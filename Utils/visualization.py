import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
def visualize_data():
    """
    Visualize entity distribution and lengths in the annotated resume data.
    """
    # Path to your annotated resume data
    DATA_PATH = r"C:\ML\CV-Parsing\Data\Entity Recognition in Resumes.json"

    # Load data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Extract entities
    entity_counts = Counter()
    entity_lengths = []
    for item in data:
        for annotation in item.get("annotation", []):
            label = annotation.get("label")
            if isinstance(label, list):
                if not label:
                    continue  # skip if label list is empty
                label = label[0]
            if not isinstance(label, str):
                continue  # skip if label is not a string
            for point in annotation.get("points", []):
                start = point.get("start")
                end = point.get("end")
                if start is not None and end is not None:
                    entity_counts[label] += 1
                    entity_lengths.append(end - start)

    # Plot entity type distribution
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()), palette="viridis")
    plt.title("Entity Type Distribution")
    plt.ylabel("Count")
    plt.xlabel("Entity Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot entity length distribution
    plt.figure(figsize=(10,6))
    sns.histplot(entity_lengths, bins=30, kde=True, color="skyblue")
    plt.title("Entity Length Distribution (in characters)")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Show some random annotated samples
    import random
    print("\nSample annotated texts:")
    for item in random.sample(data, min(3, len(data))):
        print("Text:", item["content"][:200], "...")
        print("Entities:", [(ann["label"], ann["points"]) for ann in item["annotation"]])
        print()
