import numpy as np
import os
import pickle
import csv

def process_data(input_filepath):
    """
    Processes the input CSV file and returns a data matrix along with a list of corrupted rows.
    """
    print(f"Starting to process CSV file: {input_filepath}")
    temp_data = []
    corrupted_rows = []

    try:
        with open(input_filepath, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader, None)

            for i, row in enumerate(reader):
                if len(row) < 4:
                    corrupted_rows.append((i, row))
                    continue
                
                user_str, product_str, rating_str, timestamp_str = row

                try:
                    temp_data.append([user_str, product_str, float(rating_str), int(timestamp_str)])
                except ValueError:
                    corrupted_rows.append((i, row))
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
        return None, None, None, []
    
    data_matrix = np.array(temp_data, dtype=object)
    print(f"Raw Data shape: {data_matrix.shape}")
    return data_matrix, corrupted_rows


def save_processed_data(data_matrix, user_map, product_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if data_matrix is not None:
        data_filepath = os.path.join(output_dir, "data_processed.npy")
        np.save(data_filepath, data_matrix)
        print(f"Saved processed data matrix to {data_filepath}")
    else:
        print("No data matrix to save.")

    if user_map is not None:
        with open(os.path.join(output_dir, "user_map.pkl"), "wb") as f:
            pickle.dump(user_map, f)
        print("Saved user_map.pkl")
    
    if product_map is not None:
        with open(os.path.join(output_dir, "product_map.pkl"), "wb") as f:
            pickle.dump(product_map, f)
        print("Saved product_map.pkl")

def load_processed_data(input_dir):
    data_filepath = os.path.join(input_dir, "data_processed.npy")

    data_matrix = np.load(data_filepath, allow_pickle=True)

    return data_matrix

def filter_k_core_iterative(data, k=5):
    """
    Optimized K-Core Filtering using Integer Encoding and Bincount.
    Speed up: ~50x faster than using np.unique on object arrays.
    """
    
    # ENCODE STRING -> INT 
    _, u_idx = np.unique(data[:, 0], return_inverse=True)
    _, i_idx = np.unique(data[:, 1], return_inverse=True)
    
    original_indices = np.arange(len(data))
    
    # Working buffers
    cur_u = u_idx
    cur_i = i_idx
    cur_idx = original_indices
    
    iteration = 0
    
    while True:
        iteration += 1
        start_len = len(cur_u)
        if len(cur_u) == 0: break
        
        u_counts = np.bincount(cur_u)
        u_keep_mask = u_counts >= k
        valid_mask_u = u_keep_mask[cur_u]
        
        cur_u = cur_u[valid_mask_u]
        cur_i = cur_i[valid_mask_u]
        cur_idx = cur_idx[valid_mask_u]
        
        if len(cur_i) == 0: break
            
        i_counts = np.bincount(cur_i)
        i_keep_mask = i_counts >= k
        
        valid_mask_i = i_keep_mask[cur_i]
        
        cur_u = cur_u[valid_mask_i]
        cur_i = cur_i[valid_mask_i]
        cur_idx = cur_idx[valid_mask_i]
        
        end_len = len(cur_u)
        print(f"   Iteration {iteration}: Reduced from {start_len} to {end_len} rows")
        
        if start_len == end_len:
            print("   Convergence reached.")
            break
    return data[cur_idx]

def build_maps_and_convert_to_int(data):
    # Create User Map 
    unique_users = np.unique(data[:, 0])
    user_map = {u_str: i for i, u_str in enumerate(unique_users)}
    
    # Create Product Map (from unique string IDs)
    unique_products = np.unique(data[:, 1])
    product_map = {p_str: i for i, p_str in enumerate(unique_products)}
    
    print(f"Maps created. Total Users: {len(user_map)}, Total Products: {len(product_map)}")
    print("Mapping data to integer matrix...")
    
    # Map Data (Convert String -> Int)
    user_indices = np.array([user_map[u] for u in data[:, 0]], dtype=np.int32)
    product_indices = np.array([product_map[p] for p in data[:, 1]], dtype=np.int32)
    
    # Cast Rating and Timestamp to float32
    ratings = data[:, 2].astype(np.float32)
    timestamps = data[:, 3].astype(np.float32)
    
    # Stack into final matrix (N, 4)
    final_data = np.column_stack((user_indices, product_indices, ratings, timestamps))
    
    print(f"Conversion complete. Final Data Shape: {final_data.shape}")
    
    return final_data, user_map, product_map

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

def perform_hypothesis_test(data, year_a=2013, year_b=2014, confidence_level=0.95):
    """Performs Z-Test comparing ratings between two years."""
    print(f"Performing Z-Test: Ratings in {year_b} > {year_a}?")
    print(f"Hypothesis: H0: Mean_B <= Mean_A | H1: Mean_B > Mean_A")
    print(f"Confidence Level: {confidence_level * 100}%")

    z_critical_table = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326
    }

    if confidence_level not in z_critical_table:
        print(f"Error: Confidence level {confidence_level} is not supported in NumPy-only mode.")
        print("Supported levels: 0.90, 0.95, 0.99")
        return
    z_critical = z_critical_table[confidence_level]
    
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
    if pooled_se == 0:
        print("Error: Pooled standard error is zero, cannot compute Z-score.")
        return

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

    return standardized_data, mean_rating, std_rating

def split_train_test(data, train_ratio=0.8):
    """Splits data based on Timestamp."""
    sorted_indices = np.argsort(data[:, 3])
    data_sorted = data[sorted_indices]

    num_train = int(len(data) * train_ratio)
    train_data = data_sorted[:num_train]
    test_data = data_sorted[num_train:]

    return train_data, test_data
