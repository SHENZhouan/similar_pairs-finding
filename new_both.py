import pandas as pd

# File paths
validation_file = "cleaned_preprocessed_validation.csv"
test_file = "cleaned_preprocessed_test.csv"
output_file = "combined_cleaned_preprocessed.csv"

# Read the two CSV files
validation_df = pd.read_csv(validation_file)
test_df = pd.read_csv(test_file)

# Append the data from test_df to the end of validation_df
combined_df = pd.concat([validation_df, test_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"Data successfully merged and saved to {output_file}")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from the combined cleaned and preprocessed CSV file
df = pd.read_csv('combined_cleaned_preprocessed.csv')

# Assuming the file contains a 'tokens' column where each cell is a comma-separated string
# Convert the strings in the 'tokens' column to lists
df['tokens'] = df['tokens'].apply(lambda x: x.split(','))

# Step 1: Convert the list of tokens into cleaned text (space-separated string)
df['final'] = df['tokens'].apply(lambda x: ' '.join(x))

# Step 2: Calculate the TF-IDF weight matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['final'])

# Step 3: Generate f-bit signature
f = 64  # Set the dimension of the f-bit signature
np.random.seed(42)  # Set a fixed random seed for reproducibility
random_vectors = np.random.choice([-1, 1], size=(tfidf_matrix.shape[1], f))  # Each feature corresponds to an f-dimensional vector

# Calculate the weighted projection vector
weighted_projection = tfidf_matrix @ random_vectors  # (n_docs, n_features) x (n_features, f)

# Convert the weighted results to 0/1 vectors
binary_signatures = (weighted_projection > 0).astype(int)  # If greater than 0, assign 1; otherwise, assign 0

# Step 4: Save the results to a file
output_file = 'document_signatures_combined.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("doc_id,signature\n")
    for doc_id, signature in enumerate(binary_signatures):
        signature_str = ''.join(map(str, signature))  # Convert the 0/1 vector to a string
        f.write(f"{doc_id},{signature_str}\n")

print(f"Document signatures saved to {output_file}")
import numpy as np
import pandas as pd
import torch  # For GPU acceleration

# ----------------------------------
# 1. Read the 01 vector file
# ----------------------------------
input_file = "document_signatures_combined.csv"  # Path to the 01 vector file

# Read the CSV file using pandas
df = pd.read_csv(input_file, header=None, names=["doc_id", "signature"])  # The first column is doc_id, the second is the 01 string

# Filter out rows containing characters other than 0 and 1
df = df[df["signature"].str.match(r'^[01]+$')]

# Convert the 01 strings to NumPy arrays
vectors = np.array([list(map(int, list(sig))) for sig in df["signature"]])
doc_ids = df["doc_id"].values  # Extract doc_id

# If a GPU is available, transfer the data to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vectors = torch.tensor(vectors, device=device)

# ----------------------------------
# 2. Banding
# ----------------------------------
def lsh_band_split_with_doc_ids(vectors, doc_ids, b, r):
    """
    Splits the 01 vectors into b bands, each containing r bits, while keeping the doc_id.
    :param vectors: Input 01 vector matrix (torch.Tensor)
    :param doc_ids: Corresponding document IDs (numpy array)
    :param b: Number of bands
    :param r: Number of rows per band
    :return: Segmented bands, each band contains a list of (doc_id, band_bits)
    """
    n, f = vectors.shape  # n is the number of vectors, f is the vector length
    assert f == b * r, "Vector length must be equal to b * r"

    bands = []
    for i in range(b):
        # Extract the bits for each band
        band_bits = vectors[:, i * r:(i + 1) * r]
        band = [(doc_id, band_bits[j].cpu().numpy()) for j, doc_id in enumerate(doc_ids)]
        bands.append(band)
    return bands

# Set parameters
b = 4  # Number of bands
r = 16  # Number of rows per band
bands = lsh_band_split_with_doc_ids(vectors, doc_ids, b, r)

# Print the shape of the first band and some examples
print(f"Band 0 shape: {len(bands[0])} documents, each with {r} bits")
print(f"Example from Band 0: {bands[0][:3]}")  # Print the first 3 (doc_id, band_bits) from Band 0
from tqdm import tqdm

# ----------------------------------
# 3. Hashing each band
# ----------------------------------
def hash_band(band):
    """
    Hashes each vector in the band while retaining the doc_id.
    :param band: List of elements in the band, each element is (doc_id, array([...]))
    :return: List of hash values, each element is (doc_id, hash_value)
    """
    return [(doc_id, hash(tuple(vector))) for doc_id, vector in band]

# Hash each band
hashed_bands = [hash_band(band) for band in tqdm(bands, desc="Hashing bands")]

# Print the hash results for the first band
print(f"Hashed Band 0: {hashed_bands[0]}")

# Save hashed bands to a file
with open("hashed_bands_combined.txt", "w") as f:
    for band_idx, band_hashes in enumerate(hashed_bands):
        f.write(f"Band {band_idx}:\n")
        for doc_id, hash_value in band_hashes:
            f.write(f"{doc_id},{hash_value}\n")
        f.write("\n")
print("Hashed bands saved to hashed_bands_combined.txt")
from collections import defaultdict
import csv

def read_hashed_bands(file_path):
    """
    Reads hashed_bands data from a file and reconstructs the variable.
    :param file_path: Path to the file
    :return: List of hashed_bands, where each element is [(doc_id, hash_value), ...]
    """
    hashed_bands = []
    with open(file_path, "r", encoding="utf-8") as f:
        current_band = []
        for line in f:
            line = line.strip()
            if line.startswith("Band"):
                # If a new Band is encountered, save the current Band and start a new one
                if current_band:
                    hashed_bands.append(current_band)
                    current_band = []
            elif line:  # Non-empty line
                doc_id, hash_value = line.split(",")
                current_band.append((int(doc_id), int(hash_value)))
        # Add the last Band
        if current_band:
            hashed_bands.append(current_band)
    return hashed_bands

def build_hash_buckets(hashed_bands):
    """
    Builds hash buckets based on the hash values.
    :param hashed_bands: List of hash values for each band, where each band is [(doc_id, hash_value), ...]
    :return: Hash buckets
    """
    buckets = defaultdict(list)
    for band_idx, band_hashes in enumerate(hashed_bands):
        for doc_id, hash_value in band_hashes:
            # Use (band_idx, hash_value) as the key and add the doc_id to the corresponding bucket
            buckets[(band_idx, hash_value)].append(doc_id)
    return buckets

def save_hash_buckets_to_csv(hash_buckets, output_file):
    """
    Saves the hash_buckets dictionary to a CSV file.
    :param hash_buckets: Hash bucket dictionary, keys are (band_idx, hash_value), values are lists of document IDs
    :param output_file: Path to the output CSV file
    """
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Band_Index", "Hash_Value", "Document_Indices"])

        # Write the contents of each hash bucket
        for (band_idx, hash_value), doc_ids in hash_buckets.items():
            writer.writerow([band_idx, hash_value, doc_ids])

# Use the function to read hashed_bands
file_path = "hashed_bands_combined.txt"  # Replace with your file path
hashed_bands = read_hashed_bands(file_path)

# Build hash buckets
hash_buckets = build_hash_buckets(hashed_bands)

# Print a few results to verify
for key, value in list(hash_buckets.items())[:5]:  # Print the first 5 buckets
    print(f"Bucket {key}: {value}")

# Save hash_buckets to a CSV file
output_file = "hash_buckets_combined.csv"  # Replace with your output file path
save_hash_buckets_to_csv(hash_buckets, output_file)

print(f"Hash buckets saved to {output_file}")
import csv
import numpy as np
from itertools import combinations
from tqdm import tqdm

def load_binary_vectors(file_path):
    """
    Loads binary vectors from a file.
    :param file_path: Path to the binary vector file
    :return: Dictionary of binary vectors {doc_id: vector}
    """
    try:
        binary_vectors = {}
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                doc_id = int(row[0])  # First column is doc_id
                vector = np.array(list(map(int, row[1])))  # Second column is the 01 string
                binary_vectors[doc_id] = vector
        return binary_vectors
    except Exception as e:
        print(f"Error loading binary vector file: {e}")
        raise

def hamming_distance(vec1, vec2):
    """
    Calculates the Hamming distance between two binary vectors.
    :param vec1: The first binary vector
    :param vec2: The second binary vector
    :return: The Hamming distance
    """
    xor_dist = np.bitwise_xor(vec1, vec2)
    return np.count_nonzero(xor_dist)

def find_similar_pairs(bands_file, vectors_file, output_file, threshold=10):
    """
    Finds similar pairs based on bands and binary vectors.
    :param bands_file: Path to the bands file
    :param vectors_file: Path to the binary vector file
    :param output_file: Path to save the similar pairs
    :param threshold: Hamming distance threshold
    """
    # Load binary vectors
    print("Loading binary vectors...")
    binary_vectors = load_binary_vectors(vectors_file)

    # Load bands data
    print("Loading bands data...")
    bands = []
    with open(bands_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            band_index = int(row[0])  # First column is band_index
            hash_value = int(row[1])  # Second column is hash_value
            vector_indices = eval(row[2])  # Third column is the list of document indices
            bands.append(vector_indices)
    candidate_num = 0
    # Find similar pairs
    print("Calculating similar pairs...")
    similar_pairs = set()
    for group in tqdm(bands, desc="Processing bands"):
        if len(group) > 1:
            for idx1, idx2 in combinations(group, 2):
                if idx1 == idx2:
                    continue
                elif abs(idx1 - idx2) < 163597:
                    continue
                else:
                    candidate_num += 1
                    vec1 = binary_vectors[idx1]
                    vec2 = binary_vectors[idx2]
                    dist = hamming_distance(vec1, vec2)
                    if dist <= threshold:
                        similar_pairs.add((idx1, idx2, dist))
    print(f"Number of candidate pairs found: {candidate_num}")
    # Save similar pairs to a file
    print("Saving similar pairs to file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Number of similar pairs found: {len(similar_pairs)}\n")
        for pair in sorted(similar_pairs):
            f.write(f"{pair[0]},{pair[1]},Hamming Distance: {pair[2]}\n")

    print(f"Similar pairs saved to {output_file}")

# Example usage
bands_file = "hash_buckets_combined.csv"  # Path to the bands file
vectors_file = "document_signatures_combined.csv"  # Path to the binary vector file
output_file = "similar_pairs_combined.csv"  # Path to the output file
threshold = 4  # Hamming distance threshold

find_similar_pairs(bands_file, vectors_file, output_file, threshold)
import pandas as pd

# File paths
similar_pairs_file = "similar_pairs_combined.csv"
processed_test_file = "combined_cleaned_preprocessed.csv"
output_file = "similar_pairs_with_texts_combined.csv"

# Read the similar_pairs_test.csv file
similar_pairs = pd.read_csv(similar_pairs_file, header=None, names=["Index1", "Index2", "Hamming_Distance"], skiprows=1)

# Read the preprocessed_test.csv file
processed_test = pd.read_csv(processed_test_file)

# Fill NaN values in the cleaned_text column with empty strings
processed_test["cleaned_text"] = processed_test["cleaned_text"].fillna("")

# Create a new DataFrame to store the results
output_data = []


# Iterate through each similar pair
for _, row in similar_pairs.iterrows():
    index1 = int(row["Index1"])
    index2 = int(row["Index2"])
    hamming_distance = row["Hamming_Distance"]

    # Get the corresponding text content
    text1 = processed_test.loc[index1, "cleaned_text"]  # Note: index starts from 0
    text2 = processed_test.loc[index2, "cleaned_text"]

    # Add to the output data
    output_data.append({
        "Index1": index1,
        "Index2": index2,
        "Hamming_Distance": hamming_distance,
        "Text1": text1,
        "Text2": text2
    })


# Save the results to a new CSV file
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Text content of similar pairs saved to {output_file}")
import pandas as pd

# File path
file_path = "similar_pairs_combined.csv"
df = pd.read_csv(file_path)
df = pd.read_csv(file_path, header=None, names=["val", "test", "hamming_distance"])  # Specify column names
# Skip non-data rows (e.g., "找到的相似对数量")
df = df[~df["val"].str.contains("找到的相似对数量", na=False)]




df["test"] = df["test"] - 163597  # Subtract the offset
df["test"] = df["test"].astype(int)  # Convert to integer type

# Print the transformed DataFrame
print(df.head())

# Save back to file (optional)
df.to_csv("cleaned_similar_pairs_combined.csv", index=False)
import pandas as pd

# File paths
similar_pairs_file = "cleaned_similar_pairs_combined.csv"
validation_file = "cleaned_preprocessed_validation.csv"
test_file = "cleaned_preprocessed_test.csv"
output_file = "similar_pairs_with_texts1.csv"

# Read the similar_pairs.csv file
similar_pairs = pd.read_csv(similar_pairs_file)

# Read the cleaned_preprocessed_validation.csv and cleaned_preprocessed_test.csv files
validation_data = pd.read_csv(validation_file)
test_data = pd.read_csv(test_file)

# Ensure no missing values in the cleaned_text column
validation_data["cleaned_text"] = validation_data["cleaned_text"].fillna("")
test_data["cleaned_text"] = test_data["cleaned_text"].fillna("")

# Create a new DataFrame to store the results
output_data = []

# Iterate through each similar pair
for index, row in similar_pairs.iterrows():


    val_doc_id = row["val"]
    test_doc_id = row["test"]
    hamming_distance = row["hamming_distance"]


    # Get the corresponding text content
    # Corrected code

    val_text = validation_data.iloc[val_doc_id]["cleaned_text"]
    test_text = test_data.iloc[test_doc_id]["cleaned_text"]
    # Add to the output data
    output_data.append({
        "Validation_Doc_ID": val_doc_id,
        "Test_Doc_ID": test_doc_id,
        "Hamming_Distance": hamming_distance,
        "Validation_Text": val_text,
        "Test_Text": test_text
    })

# Save the results as a DataFrame
output_df = pd.DataFrame(output_data)

# Save the results to a new CSV file
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Text content of similar pairs saved to {output_file}")
