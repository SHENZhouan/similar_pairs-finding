import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import mmh3
import os
from itertools import combinations
import ast
from tqdm import tqdm  # For progress bars

class MinHashLSH:
    def __init__(self, num_perm=128, num_bands=32, random_seed=None):
        """
        Initialize MinHashLSH with specified parameters
        
        Args:
            num_perm: Number of permutation functions (hash functions)
            num_bands: Number of bands for LSH
            random_seed: Seed for deterministic hash function generation
        """
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.random_seed = random_seed
        self.hash_funcs = self._create_hash_functions()
        
    def _create_hash_functions(self):
        """Generate a deterministic family of hash functions"""
        rng = np.random.RandomState(self.random_seed)
        return [(rng.randint(1, 2**32), rng.randint(1, 2**32)) 
                for _ in range(self.num_perm)]
    
    def _minhash(self, shingles):
        """
        Generate MinHash signature for a set of shingles
        """
        signature = np.full(self.num_perm, np.inf)
        
        for shingle in shingles:
            for i, (a, b) in enumerate(self.hash_funcs):
                hash_val = (a * mmh3.hash(str(shingle)) + b) % (2**32 - 1)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        return signature
    
    def _lsh_bands(self, signature):
        """Split signature into bands for LSH"""
        bands = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = (i + 1) * self.rows_per_band
            band = signature[start:end]
            band_hash = mmh3.hash_bytes(band.tobytes())
            bands.append(band_hash)
        return bands
    
    def fit_transform(self, documents, use_tokens=True):
        """Process documents with chunking support"""
        if use_tokens:
            shingle_sets = [set(doc) for doc in documents]
        else:
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
            X = vectorizer.fit_transform(documents)
            shingle_sets = [set(vectorizer.get_feature_names_out()[np.where(row.toarray()[0] > 0)])
                          for row in X]
        
        # Generate signatures with progress bar
        signatures = np.array([self._minhash(shingles) for shingles in tqdm(shingle_sets, desc="Generating signatures")])
        
        # LSH processing
        buckets = {}
        for doc_id, signature in enumerate(tqdm(signatures, desc="Processing LSH bands")):
            bands = self._lsh_bands(signature)
            for band_id, band_hash in enumerate(bands):
                key = (band_id, band_hash)
                buckets.setdefault(key, []).append(doc_id)
        
        # Generate candidate pairs
        candidate_pairs = set()
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    candidate_pairs.add(tuple(sorted(pair)))
        
        return {
            'signatures': signatures,
            'buckets': buckets,
            'candidate_pairs': list(candidate_pairs)
        }

def load_data_in_chunks(validation_path, test_path, chunk_size=10000):
    """Load data in chunks to reduce memory usage"""
    def process_chunk(chunk):
        chunk['tokens'] = chunk['tokens'].apply(ast.literal_eval)
        return chunk
    
    # Read validation data
    val_chunks = []
    for chunk in pd.read_csv(validation_path, chunksize=chunk_size):
        val_chunks.append(process_chunk(chunk))
    val_df = pd.concat(val_chunks)
    
    # Read test data
    test_chunks = []
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        test_chunks.append(process_chunk(chunk))
    test_df = pd.concat(test_chunks)
    
    return val_df, test_df

def jaccard_similarity(set1, set2):
    """Calculate actual Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def evaluate_results(candidate_pairs, signatures, val_docs, test_docs, threshold=0.8):
    """Evaluate pairs using actual Jaccard similarity"""
    verified_pairs = []
    similarity_scores = []
    
    for i, j in tqdm(candidate_pairs, desc="Evaluating pairs"):
        # Determine which set each document comes from
        if i < len(val_docs):
            set_i = set(val_docs[i])
            origin_i = 'validation'
            orig_id_i = i
        else:
            set_i = set(test_docs[i - len(val_docs)])
            origin_i = 'test'
            orig_id_i = i - len(val_docs)
            
        if j < len(val_docs):
            set_j = set(val_docs[j])
            origin_j = 'validation'
            orig_id_j = j
        else:
            set_j = set(test_docs[j - len(val_docs)])
            origin_j = 'test'
            orig_id_j = j - len(val_docs)
        
        # Calculate true Jaccard similarity
        similarity = jaccard_similarity(set_i, set_j)
        similarity_scores.append(similarity)
        
        if similarity >= threshold:
            verified_pairs.append((orig_id_i, orig_id_j, similarity, origin_i, origin_j))
    
    return {
        'verified_pairs': verified_pairs,
        'similarity_scores': similarity_scores
    }

def save_near_duplicates(evaluation, val_docs, test_docs, output_path="near_duplicates.csv"):
    """Save all near-duplicates to CSV with chunking support"""
    rows = []
    
    for orig_id_i, orig_id_j, sim, origin_i, origin_j in tqdm(evaluation['verified_pairs'], desc="Preparing duplicates"):
        # Get the appropriate document
        doc_i = val_docs[orig_id_i] if origin_i == 'validation' else test_docs[orig_id_i]
        doc_j = val_docs[orig_id_j] if origin_j == 'validation' else test_docs[orig_id_j]
        
        # Get text previews
        preview_i = ' '.join(doc_i[:50]) if len(doc_i) > 0 else "<EMPTY>"
        preview_j = ' '.join(doc_j[:50]) if len(doc_j) > 0 else "<EMPTY>"
        
        rows.append({
            'doc_id_1': orig_id_i,
            'doc_id_2': orig_id_j,
            'similarity': sim,
            'set_1': origin_i,
            'set_2': origin_j,
            'preview_1': preview_i,
            'preview_2': preview_j
        })
    
    # Save in chunks to handle large outputs
    chunk_size = 10000
    for i in range(0, len(rows), chunk_size):
        chunk = pd.DataFrame(rows[i:i+chunk_size])
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
    
    print(f"\nSaved {len(rows)} near-duplicates to {output_path}")

def main():
    # Load data in chunks
    print("Loading data...")
    val_df, test_df = load_data_in_chunks(
        'cleaned_preprocessed_test.csv',
        'cleaned_preprocessed_validation.csv',
        chunk_size=25000  # Adjust based on your memory
    )
    
    # Keep documents separate
    val_docs = val_df['tokens'].tolist()
    test_docs = test_df['tokens'].tolist()
    
    # Combine for processing (but track original indices)
    all_docs = val_docs + test_docs
    
    # Initialize with fixed random seed for reproducibility
    minhash_lsh = MinHashLSH(num_perm=128, num_bands=32, random_seed=42)
    
    # Process documents
    print("\nProcessing documents...")
    results = minhash_lsh.fit_transform(all_docs, use_tokens=True)
    
    # Evaluate results
    print("\nEvaluating candidate pairs...")
    evaluation = evaluate_results(
        results['candidate_pairs'],
        results['signatures'],
        val_docs,
        test_docs
    )
    
    # Save results
    print("\nSaving results...")
    save_near_duplicates(evaluation, val_docs, test_docs)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Validation documents: {len(val_docs)}")
    print(f"Test documents: {len(test_docs)}")
    print(f"Total documents: {len(val_docs) + len(test_docs)}")
    print(f"Candidate pairs: {len(results['candidate_pairs'])}")
    print(f"Near-duplicates found: {len(evaluation['verified_pairs'])}")

if __name__ == "__main__":
    main()