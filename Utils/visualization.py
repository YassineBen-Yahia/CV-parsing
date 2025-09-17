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
    DATA_PATH = r"C:\ML\CV-Parsing\Data\augmented_train_data.json"

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

    # Show the most frequent value for each entity type
    from collections import defaultdict
    entity_value_counts = defaultdict(Counter)
    for item in data:
        text = item.get("content", "")
        for annotation in item.get("annotation", []):
            label = annotation.get("label")
            if isinstance(label, list):
                if not label:
                    continue
                label = label[0]
            if not isinstance(label, str):
                continue
            for point in annotation.get("points", []):
                start = point.get("start")
                end = point.get("end")
                if start is not None and end is not None:
                    value = text[start:end].strip()
                    if value:
                        entity_value_counts[label][value] += 1

    # Prepare data for plotting
    labels = []
    values = []
    counts = []
    for label, counter in entity_value_counts.items():
        most_common = counter.most_common(1)
        if most_common:
            value, count = most_common[0]
            labels.append(label)
            values.append(value)
            counts.append(count)

    # Plot most frequent value for each entity type
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels, y=counts, palette="mako")
    plt.title("Most Frequent Value Count for Each Entity Type")
    plt.ylabel("Count")
    plt.xlabel("Entity Type")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate bars with the most frequent value
    for i, (v, c) in enumerate(zip(values, counts)):
        plt.text(i, c, f"'{v}'", ha='center', va='bottom', fontsize=8, rotation=90)

    plt.show()

    

if __name__ == "__main__":
    visualize_data()
