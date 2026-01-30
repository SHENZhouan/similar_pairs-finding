import csv
from collections import defaultdict

# Initialize a dictionary to store the counts
pair_counts = defaultdict(int)

# Filepath to the CSV file
file_path = "bit_sampling_similar_pairs_try2.csv"

# Open and read the CSV file
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Iterate through each row in the CSV
    for row in reader:
        # Check if both sources are from the test set
        if row['source_1'] == 'test' and row['source_2'] == 'test':
            # Create a tuple for the (doc_id_1, doc_id_2) pair
            pair = (row['doc_id_1'], row['doc_id_2'])
            
            # Increment the count for this pair
            pair_counts[pair] += 1
# Filepath to the output CSV file
output_file_path = "pair_counts_output_test.csv"

# Write the results to a new CSV file
with open(output_file_path, mode='w', encoding='utf-8', newline='') as output_file:
    writer = csv.writer(output_file)
    
    # Write the header
    writer.writerow(["doc_id_1", "doc_id_2", "count"])
    
    # Write each pair and its count
    for (doc_id_1, doc_id_2), count in pair_counts.items():
        writer.writerow([doc_id_1, doc_id_2, count])

print(f"Results have been written to {output_file_path}.")





import csv
from collections import defaultdict

# Initialize a dictionary to store the counts
pair_counts = defaultdict(int)

# Load existing pair counts from pair_counts_output_test.csv
pair_counts_file = "pair_counts_output_test.csv"
with open(pair_counts_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        pair = (row['doc_id_1'], row['doc_id_2'])
        pair_counts[pair] = int(row['count'])

# Process near_duplicates(1).csv to update pair_counts
near_duplicates_file = "near_duplicates.csv"
with open(near_duplicates_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Check if both sources are 'test'
        if row['set_1'] == 'test' and row['set_2'] == 'test':
            pair = (row['doc_id_1'], row['doc_id_2'])
            pair_counts[pair] += 1  # Increment count if exists, or set to 1 if new

# Write updated pair_counts back to pair_counts_output.csv
with open(pair_counts_file, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["doc_id_1", "doc_id_2", "count"])  # Write header
    for (doc_id_1, doc_id_2), count in pair_counts.items():
        writer.writerow([doc_id_1, doc_id_2, count])

print(f"Updated pair counts have been written to {pair_counts_file}.")





import csv
from collections import defaultdict

# Initialize a dictionary to store the counts
pair_counts = defaultdict(int)

# Load existing pair counts from pair_counts_output_test.csv
pair_counts_file = "pair_counts_output_test.csv"
with open(pair_counts_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        pair = (row['doc_id_1'], row['doc_id_2'])
        pair_counts[pair] = int(row['count'])

# Process similar_pairs_test.csv to update pair_counts
similar_pairs_file = "similar_pairs_test.csv"
with open(similar_pairs_file, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the first line (header or metadata)
    for row in reader:
        # Extract doc_id_1 and doc_id_2 from the first two columns
        doc_id_1, doc_id_2 = row[0], row[1]
        pair = (doc_id_1, doc_id_2)
        pair_counts[pair] += 1  # Increment count if exists, or set to 1 if new

# Write updated pair_counts back to pair_counts_output_test.csv
with open(pair_counts_file, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["doc_id_1", "doc_id_2", "count"])  # Write header
    for (doc_id_1, doc_id_2), count in pair_counts.items():
        writer.writerow([doc_id_1, doc_id_2, count])

print(f"Updated pair counts have been written to {pair_counts_file}.")





import csv

# Filepath to the pair_counts_output_test.csv file
pair_counts_file = "pair_counts_output_test.csv"
output_file = "filtered_pair_counts_test.csv"

# Initialize a list to store keys with values >= 2
keys_with_high_values = []

# Read the pair_counts_output_test.csv file
with open(pair_counts_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Check if the count is >= 2
        if int(row['count']) >= 2:
            keys_with_high_values.append((row['doc_id_1'], row['doc_id_2'], row['count']))

# Write the filtered results to a new CSV file
with open(output_file, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["doc_id_1", "doc_id_2", "count"])  # Write header
    writer.writerows(keys_with_high_values)

print(f"Filtered results have been written to {output_file}.")