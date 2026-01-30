# Fast & Scalable Text Deduplication
## 1. Environment
、bash
conda create -n dedup python=3.11
conda activate dedup
pip install -r requirements.txt
conda env create -f environment.yml



2. Full Pipeline
# 1) preprocess (x6 min，peak RAM ~1 GB)

python run data preprocess.py
# output the preprocessed_test.csv and preprocessed_validation.csv

python run data_preprocess_improved.py
# output the cleaned_preprocessed_test.csv and cleaned_preprocessed_validation.csv


# 2) three fingerprint engines

python minhash_implementation.py
# output near_candidates.csv by applying MinHash

python run new_both.py
# output cleaned_similar_pairs_combined.csv by applying SimHash
python run new_test.py
# output similar_pairs_test.csv by applying SimHash
python run new_validation.py
# output similar_pairs_validation.csv by applying SimHash

python run bitsampling.py
# outputs bit_sampling_similar_pairs_try2.csv by applying Bit Sampling



# 3) evaluation (precision/recall,runtime， RAM)
python run voting_both.py
# outputs pair_counts_output_both.csv and filtered_pair_counts_both.csv

python run voting_test.py
# outputs pair_counts_output_test.csv and filtered_pair_counts_test.csv

python run voting_validation.py
# outputs pair_counts_output_validation.csv and filtered_pair_counts_validation.csv



File Structure
Project1_codes/
├── data preprocess.py
├── data_preprocess_improved.py
├── cleaned_preprocessed_test.csv            
├── cleaned_preprocessed_validation.csv
│
├── minhash_implementation.py
├── new_both.py
├── new_test.py
├── new_validation.py
├── bitSampling1.py
│
├── voting_both.py
├── voting_test.py
├── voting_validation.py
│
└── project_codes.md



Output Tree
├── preprocessed_test.csv
├── preprocessed_validation.csv
├── cleaned_preprocessed_test.csv            
├── cleaned_preprocessed_validation.csv
│
├── near_candidates.csv
├── cleaned_similar_pairs_combined.csv
├── similar_pairs_test.csv
├── similar_pairs_validation.csv
├── bit_sampling_similar_pairs_try2.csv
│
├── pair_counts_output_both.csv
├── filtered_pair_counts_both.csv
├── pair_counts_output_test.csv
├── filtered_pair_counts_test.csv
├── pair_counts_output_validation.csv
└── filtered_pair_counts_validation.csv



