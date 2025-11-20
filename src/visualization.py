import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rating_distribution(data_matrix, title="Distribution of Ratings"):
    ratings = data_matrix[:, 2]  # Assuming ratings are in the third column
    values, counts = np.unique(ratings, return_counts=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=values, y=counts, palette="viridis")
    plt.title(title)
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_long_tail(data, column_index=1, entity_name="Products"):
    plt.figure(figsize=(10, 6))

    ids = data[:, column_index]

    _, counts = np.unique(ids, return_counts=True)

    sorted_counts = np.sort(counts)[::-1]

    plt.plot(sorted_counts, color="skyblue", linewidth=2)
    plt.title(f"Long-Tail Distribution of {entity_name} Ratings")
    plt.xlabel(f"{entity_name} Index (Sorted by Popularity)")
    plt.ylabel("Number of Ratings")
    plt.fill_between(range(len(sorted_counts)), sorted_counts, color="skyblue", alpha=0.4)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()