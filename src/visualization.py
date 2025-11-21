import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rating_distribution(data):
    plt.figure(figsize=(10, 6))
    
    ratings = data[:, 2]
    unique_ratings, counts = np.unique(ratings, return_counts=True)
    
    # Color palette for ratings 1-5
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#0275d8', '#5cb85c']
    
    ax = sns.barplot(x=unique_ratings, y=counts, palette=colors)
    
    plt.title("Distribution of Ratings", fontsize=16)
    plt.xlabel("Rating Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i, v in enumerate(counts):
        ax.text(i, v + (max(counts)*0.01), str(int(v)), ha="center", fontweight="bold")
        
    plt.show()

def plot_long_tail(data, column_index=1, entity_name="Products"):
    plt.figure(figsize=(10, 6))

    ids = data[:, column_index]

    _, counts = np.unique(ids, return_counts=True)

    sorted_counts = np.sort(counts)[::-1]

    plt.plot(sorted_counts, color="skyblue", linewidth=2)
    plt.fill_between(range(len(sorted_counts)), sorted_counts, color="skyblue", alpha=0.3)
    
    plt.title(f"Long-tail Distribution of {entity_name.capitalize()} Activity", fontsize=16)
    plt.xlabel(f"{entity_name.capitalize()} Index (Sorted by Popularity)", fontsize=12)
    plt.ylabel("Number of Ratings", fontsize=12)
    
    # Use Log scale because Amazon data is usually power-law distributed
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    plt.show()

def plot_popularity_vs_quantity(data):
    plt.figure(figsize=(10, 6))

    product_ids = data[:, 1].astype(int)
    ratings = data[:, 2]
    # Calculate popularity and quality per product
    products, product_counts = np.unique(product_ids, return_counts=True)

    max_id = products.max()
    sum_ratings = np.zeros(max_id + 1)

    # sum of ratings per product (vectorization)
    np.add.at(sum_ratings, product_ids, ratings)
    # Extract only the relevant sums
    relevant_sums = sum_ratings[products]
    mean_ratings = relevant_sums / product_counts

    sns.scatterplot(x=product_counts, y=mean_ratings, color="purple", alpha=0.6, s=15)

    global_mean = np.mean(ratings)
    plt.axhline(y=global_mean, color="red", linestyle="--", label=f"Global Mean: {global_mean:.2f}", linewidth=2)

    plt.title("Correlation: Popularity (Count) vs Quality (Mean Rating)", fontsize=16)
    plt.xlabel("Number of Ratings (Popularity-Scaled)", fontsize=12)
    plt.ylabel("Average Rating (Quality)", fontsize=12)
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def plot_temporal_trends(data):
    plt.figure(figsize=(10, 6))

    timestamps = data[:, 3].astype(int)
    # convert to years
    years = 1970 + (timestamps / 31536000)

    plt.hist(years, bins=int(years.max() - years.min()), color="teal", alpha=0.7, edgecolor="white")

    plt.title("Rating Volume Over Time", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Ratings", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

def plot_user_consistency(data):
    plt.figure(figsize=(10, 6))

    user_ids = data[:, 0].astype(int)
    ratings = data[:, 2]

    global_mean = np.mean(ratings)
    # RMSE calculation
    naive_rmse = np.sqrt(np.mean((ratings - global_mean) ** 2))
    print(f"Global Mean Rating: {global_mean:.4f}")
    print(f"Baseline RMSE (Predict Global Mean): {naive_rmse:.4f}")

    users, user_counts = np.unique(user_ids, return_counts=True)
    max_user_id = user_ids.max()

    # sum of ratings per user
    sum_ratings = np.zeros(max_user_id + 1)
    np.add.at(sum_ratings, user_ids, ratings)

    # sum of squared ratings per user
    sum_squared_ratings = np.zeros(max_user_id + 1)
    np.add.at(sum_squared_ratings, user_ids, ratings ** 2)

    # Filter only users with >= 2 ratings (to calculate std dev)
    valid_indices = users[user_counts >= 2]
    valid_counts = user_counts[user_counts >= 2]

    valid_sum = sum_ratings[valid_indices]
    valid_sum_squared = sum_squared_ratings[valid_indices]

    # mean of squared ratings
    mean_squared = valid_sum_squared / valid_counts
    # Square of mean ratings
    squared_mean = (valid_sum / valid_counts) ** 2
    # Variance = Mean(X^2) - Mean(X)^2
    variance = np.abs(mean_squared - squared_mean)
    user_std_devs = np.sqrt(variance)

    plt.hist(user_std_devs, bins=30, color="#8e44ad", alpha=0.7, edgecolor="white")

    plt.title("User Rating Consistency (Standard Deviation)", fontsize=16)
    plt.xlabel("Standard Deviation of Ratings", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    
    mean_std = np.mean(user_std_devs)
    plt.axvline(x=mean_std, color="r", linestyle="--", linewidth=2, label=f"Mean Std: {mean_std:.2f}")
    
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()