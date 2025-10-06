# This code runs a local LLM on your own computer and send the questionnaire directly to it.
# Each LLM-model must be downloaded to your computer before it can be used.

# pip install ollama




"""
import libraries
"""
import pandas as pd
import os
import ollama
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse   # for reading arguments in a regular way




"""
Reading Number of running as input
"""
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True, help="Run number of the experiment")
args = parser.parse_args()
print(f"Run number: {args.run}")





"""
defin function for sending a question to the model and returning the result
"""
def query(prompt, model):
    response = ollama.chat(
        model = model,
        messages= [
            {"role" : "system" , "content" : "You are filling out a personality test."
                                             "For each question, just return a number between 1 and 10, without any additional text."
                                             "Do not add explanations, words, or multiple numbers."}, 
            {"role" : "user" , "content" : prompt}
        ],
        options={
            "temperature": 0.9
        }    
    )
    raw = response['message']['content'].strip()

    # Extract only the first number in the response
    
    match = re.search(r"\d+", raw)
    if match:
        return int(match.group())
    else:
        return 5  # fallback default if model misbehaves





"""
Load questionnaire
"""
path = 'Forms/personality_test.csv'
df = pd.read_csv(path)





"""
Store all results here
"""
results =[]
model_list = ["gemma3:4b" , "llama3.1:8b"]

for model in model_list:
    for run in range(args.run):
        print(model , run+1)
        for q in df['question_text']:
            #print(f"sending questions: {q}")
            ans = query(q, model)
            #print(f"Answer: {ans}")
            results.append({"model" : model,"run" : run+1, "question" : q, "answers" : ans})


# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
Path("results").mkdir(exist_ok=True)
output_file = "results/personality_test_results.csv"
results_df.to_csv(output_file, mode="a", header=not pd.io.common.file_exists(output_file), index=False)

print(f"Results saved to: {output_file}")



"""
If you want to read the results from the 'result.csv' file
"""
#path_results = 'results/personality_test_results.csv'
#results_df = pd.read_csv(path_results)




"""
Define short labels for questions
""" 
question_labels = {
    "I am a sociable person. (Please answer with a single number between 1 and 10)": "Sociability",
    "I usually complete tasks carefully. (Answer with a number 1 to 10)": "Carefulness",
    "I get angry easily. (Answer with a number 1 to 10)": "Emotional",
    "I am very curious to learn new things. (Answer with a number 1 to 10)": "Curiosity",
    "I usually trust other people. (Answer with a number 1 to 10)": "Trust"
}





"""
Define models list
"""
models = results_df["model"].unique()




"""
Error bar chart for each model
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)  # 1 row, 2 columns

for ax, model in zip(axes, models):
    subset = results_df[results_df["model"] == model]
    stats = subset.groupby("question")["answers"].agg(["min", "max", "mean"]).reset_index()
    
    x = range(len(stats))
    y = stats["mean"]
    yerr = [y - stats["min"], stats["max"] - y]  # min/max error bars
    
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=5, label=model, color="blue")
    ax.set_xticks(x)
    ax.set_xticklabels([question_labels[q] for q in stats["question"]], rotation=45, ha="right")
    ax.set_title(model)
    ax.set_ylabel("Answer (1–10)")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Answer Ranges per Question – Comparison of Models")
plt.ylim(0, 10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(Path("results") / "answer_for_models_run", dpi=300)




"""
Error bar chart for comparison on models
"""
# Aggregate per model and question
stats = results_df.groupby(["model", "question"])["answers"].agg(["min", "max", "mean"]).reset_index()

# Prepare data
models = stats["model"].unique()
questions = stats["question"].unique()
x = np.arange(len(questions))  # positions for questions
width = 0.25  # horizontal spacing between models

fig, ax = plt.subplots(figsize=(8, 5))

for i, model in enumerate(models):
    subset = stats[stats["model"] == model]
    
    means = subset["mean"].values
    mins = subset["min"].values
    maxs = subset["max"].values
    
    # error bars = distance from mean to min/max
    yerr = [means - mins, maxs - means]
    
    ax.errorbar(
        x + i * width, means, yerr=yerr, fmt="o", capsize=5, label=model, markersize=6
    )

# Labels and style
ax.set_xticks(x + width / 2)
ax.set_xticklabels([question_labels[q] for q in questions], rotation=30, ha="right")
ax.set_ylabel("Answer (1–10)")
ax.set_title("Model Comparison – Error Bar Chart (Min–Max Range)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.ylim(0, 10)
plt.savefig(Path("results") / "models_comparison.png", dpi=300)





"""
Radar chart
"""
stats_by_model = {}

# Compute average answers per question for each model
for model in models:
    subset = results_df[results_df["model"] == model]
    stats = subset.groupby("question")["answers"].mean().reset_index()
    stats_by_model[model] = stats

# Radar chart setup
labels = [question_labels[q] for q in stats_by_model[models[0]]["question"]]
N = len(labels)

# Repeat first value at the end to close the circle
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Plot each model
for model in models:
    values = stats_by_model[model]["answers"].tolist()
    values += values[:1]  # close the circle
    
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.25)

# Set labels and style
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 10)
plt.title("Average Personality Profile – Model Comparison", size=12, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.savefig(Path("results") / "radar_chart.png", dpi=300)
