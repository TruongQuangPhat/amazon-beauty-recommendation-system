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