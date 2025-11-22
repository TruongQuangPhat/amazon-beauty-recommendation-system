import numpy as np
import os
import pickle
import csv

def process_csv_to_numpy(input_filepath):
    print(f"Starting to process CSV file: {input_filepath}")

    user_map = {}
    product_map = {}

    current_user_index = 0
    current_product_index = 0

    temp_data = []
    corrupted_rows = []

    try:
        with open(input_filepath, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row

            for i, row in enumerate(reader):
                # # Check for corrupted rows (missing columns)
                if len(row) < 4:
                    corrupted_rows.append((i, row))
                    continue
                user_str, product_str, rating_str, timestamp_str = row

                if user_str not in user_map:
                    user_map[user_str] = current_user_index
                    current_user_index += 1
                user_index = user_map[user_str]

                if product_str not in product_map:
                    product_map[product_str] = current_product_index
                    current_product_index += 1
                product_index = product_map[product_str]

                try:
                    temp_data.append([user_index, product_index, float(rating_str), int(timestamp_str)])
                except ValueError:
                    corrupted_rows.append((i, row))
                    continue
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
        return None, None, None, []
    
    print(f"Converting data to NumPy array.")
    data_matrix = np.array(temp_data, dtype=np.float32)

    return data_matrix, user_map, product_map, corrupted_rows


def save_processed_data(data_matrix, user_map, product_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_filepath = os.path.join(output_dir, "ratings_data.npy")
    np.save(data_filepath, data_matrix)
    print(f"Saved data matrix to {data_filepath}")

    with open(os.path.join(output_dir, "user_map.pkl"), "wb") as f:
        pickle.dump(user_map, f)

    with open(os.path.join(output_dir, "product_map.pkl"), "wb") as f:
        pickle.dump(product_map, f)

    print(f"Saved user and product maps to {output_dir}")

def load_processed_data(input_dir):
    data_filepath = os.path.join(input_dir, "ratings_data.npy")
    user_map_filepath = os.path.join(input_dir, "user_map.pkl")
    product_map_filepath = os.path.join(input_dir, "product_map.pkl")

    data_matrix = np.load(data_filepath)

    with open(user_map_filepath, "rb") as f:
        user_map = pickle.load(f)

    with open(product_map_filepath, "rb") as f:
        product_map = pickle.load(f)

    return data_matrix, user_map, product_map  

def filter_k_core(data, k=5):
    filtered_data = data.copy()
    iteration = 0

    while True:
        iteration += 1
        original_shape = filtered_data.shape[0]

        user_ids, user_counts = np.unique(filtered_data[:, 0], return_counts=True)
        valid_users = user_ids[user_counts >= k]
        mask_users = np.isin(filtered_data[:, 0], valid_users)
        filtered_data = filtered_data[mask_users]

        product_ids, product_counts = np.unique(filtered_data[:, 1], return_counts=True)
        valid_products = product_ids[product_counts >= k]
        mask_products = np.isin(filtered_data[:, 1], valid_products)
        filtered_data = filtered_data[mask_products]

        current_shape = filtered_data.shape[0]
        if original_shape == current_shape:
            break
    return filtered_data

def add_time_features(data):
    """
    Adds 'Year' and 'Time_Weight'.
    Time_Weight is scaled from 0.2 to 1.0 based on recency.
    Returns: Enhanced Matrix (N, 6) -> [User, Item, Rating, Time, Year, Weight]
    """
    timestamps = data[:, 3].astype(int)
    years = 1970 + (timestamps / 31536000)

    min_time, max_time = timestamps.min(), timestamps.max()
    time_weights = 0.2 + 0.8 * (timestamps - min_time) / (max_time - min_time)

    enhanced_data = np.hstack((data, years.reshape(-1, 1), time_weights.reshape(-1, 1)))
    return enhanced_data

def perform_hypothesis_test(data, year_a=2013, year_b=2014):
    """Performs Z-Test comparing ratings between two years."""
    print(f"\nPerforming Z-Test: Ratings in {year_b} > {year_a}?")
    print("H0: Mean_B - Mean_A = 0 | H1: Mean_B - Mean_A > 0")
    
    # Year is at index 4
    years_col = data[:, 4].astype(int)
    ratings_a = data[years_col == year_a, 2]
    ratings_b = data[years_col == year_b, 2]
    
    if len(ratings_a) == 0 or len(ratings_b) == 0:
        print("Error: Not enough data for specified years.")
        return

    mean_a, var_a, n_a = np.mean(ratings_a), np.var(ratings_a), len(ratings_a)
    mean_b, var_b, n_b = np.mean(ratings_b), np.var(ratings_b), len(ratings_b)
    
    print(f"   {year_a}: Mean={mean_a:.4f}, N={n_a}")
    print(f"   {year_b}: Mean={mean_b:.4f}, N={n_b}")
    
    # Z-score calculation
    pooled_se = np.sqrt((var_b / n_b) + (var_a / n_a))
    z_score = (mean_b - mean_a) / pooled_se
    
    print(f"   Z-Score: {z_score:.4f}")
    if z_score > 1.645: 
        print("   Result: Reject H0. Statistically Significant.")
    else:
        print("   Result: Fail to reject H0.")

def standardize_ratings(data):
    ratings = data[:, 2]
    mean_rating = np.mean(ratings)
    std_rating = np.std(ratings)

    standardized_ratings = (ratings - mean_rating) / std_rating
    standardized_data = data.copy()
    standardized_data[:, 2] = standardized_ratings

    return standardized_data

def split_train_test(data, train_ratio=0.2):
    """Splits data based on Timestamp."""
    sorted_indices = np.argsort(data[:, 3])
    data_sorted = data[sorted_indices]

    num_train = int(len(data) * train_ratio)
    train_data = data_sorted[:num_train]
    test_data = data_sorted[num_train:]

    return train_data, test_data