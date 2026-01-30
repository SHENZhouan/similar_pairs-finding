# similar_pairs-finding

Code running instruction should be found in the project_codes.md

# **1.Overview:**

This repository realized the fingerprint method of Minhash, Simhash and Bit Sampling and used Banding method to efficiently decrease the range of possible similar pairs.

# **2.Introduction:**

In the era of big data, large-scale text corpora -ranging from web crawls to user-generated content - often contain extensive, nearly duplicate content, such as paraphrased sentences, reordered paragraphs, or slightly modified versions of the same document. These redundancies introduce significant challenges for machine learning pipelines, particularly in natural language processing (NLP). Near-duplicates can distort training signals, amplify overfitting by overrepresenting specific linguistic patterns, and lead to data leakage between training and evaluation splits. Recent studies have shown that rigorous deduplication of training data can substantially improve model generalization and downstream performance. However, traditional pairwise comparison methods, which exhibit prohibitive O($n^2$) time complexity, are infeasible for modern datasets comprising millions of documents.

Our approach implements three key fingerprinting methods—**MinHash**, **SimHash**, and **Bit Sampling**—to generate compact document signatures that approximate textual similarity. These signatures are integrated into LSH-based indexing structures (e.g., banding techniques for MinHash, bucketing for SimHash) to identify candidate near-duplicate pairs with subquadratic complexity. We evaluate this system on the Wiki40B English dataset, focusing on detecting duplicates both within and between validation and test subsets.

The primary contribution of this work is a practical, scalable framework for text deduplication that maintains high accuracy while dramatically reducing computational requirements compared to traditional approaches.

# **3.Related Work**

## **3.1 MinHash**
Prior work in MinHash and document deduplication includes:

• Original MinHash (Broder, 1997): Introduced the core technique for estimating Jaccard similarity using minimum hash values, providing the theoretical foundation for our implementation.

• LSH Variants (Gionis et al., 1999): Extended MinHash with Locality-Sensitive Hashing (LSH) bands, significantly improving scalability by reducing the number of required comparisons.

• Recent Optimizations: ”MinHash virtually always outperforms SimHash when the data are binary” are provided with a theoretical answer validated by experiments (Li., Shrivastava, 2014)as a helpful tool in deciding which LSH to use.

## **3.2 SimHash**
As early as 2002, researchers began exploring various similarity functions for Locality-Sensitive Hashing (LSH). Charikar (2002) introduced random hyperplane-based hash functions for vectors,a technique that facilitates the transformation of high-dimensional data into lower-dimensional representations and allows for the incorporation of weights for different data features. The practical utility of this approach was subsequently validated through large-scale empirical studies at Google by 2007. Google’s implementation of simhash on real-world datasets led to the development
of novel methodologies, such as employing Hamming distance calculations, to efficiently identify differences between documents. The detailed algorithms presented by Google for both online and batch queries highlight the increasing maturity of simhash applications. Nevertheless, areas for further refinement remain. Research conducted by Caitlin and Greg suggests that for realistic datasets, using fewer tags during key computation can enhance performance, and that the 0x00 pattern is a critical factor influencing efficiency. These findings offer potential avenues for improvement and innovation in the design of our LSH scheme.

## **3.3 Bit Sampling**

3.3.1 Theoretical Foundations and Early Explorations

The core idea of Bit Sampling originates from studies on adapting Locality-Sensitive Hashing (LSH) to the Hamming distance. Charikar (2002) first proposed using random hyperplane–based hash functions to map high-dimensional vectors into binary signatures, and proved that these hash functions satisfy the locality-sensitive property: when two vectors have a small Hamming distance, the probability that their hash values coincide increases significantly. This theoretical result laid the mathematical foundation for the subsequent design of Bit Sampling algorithms. Early applications focused primarily on text similarity detection—for example, by computing the Hamming distance between document signatures to quickly identify duplicate web pages.

3.3.2 Algorithmic Optimizations and Cross-Domain Extensions

Recent research has progressed along two main directions: precision enhancement and computational efficiency. For high-dimensional dense datasets, hybrid architectures have been proposed, such as combining Bit Sampling with p-stable distribution LSH to support Euclidean distance metrics. Moreover,
neural network–driven approaches (e.g., LLSH) replace traditional hash functions with multi-layer perceptrons, achieving up to 15% higher accuracy in image retrieval tasks compared to conventional Bit Sampling, at the expense of added hardware dependencies due to GPU acceleration.

# 4. Data Preprocessing

## 4.1 Data Loading
We load the English Wiki40B dataset (test and validation splits) using Hugging Face `datasets` library and convert it into pandas DataFrames.

## 4.2 Text Cleaning
Each document is normalized through:
- Lower-casing
- Removing HTML tags, punctuation, and extra whitespace
- Two regex rules that erase `start-article`, `start-section`, `start-paragraph` prefixes (any spacing or capitalization)
- A generic pattern `\b\w*newline\w*\b` that deletes in-line markers such as `2007newlinethe`, `guys-newlinea`, etc. The cleaned string is stored in `cleaned_text`.

## 4.3 Tokenization and Stop-word Removal
`cleaned_text` is split on whitespace, NLTK English stop-words are removed, and the remaining tokens are sorted by length so that similar documents yield an identical token order.

## 4.4 Residual-Tag Removal
As a safety net, we run one last pass over both `cleaned_text` and tokens. Any lingering layout markers—words like `start`, `section`, `paragraph`, or custom artifacts such as `crimsontoolish`—are filtered out, along with every `newline*` variant. This guarantees no formatting noise reaches the fingerprint stage.

## 4.5 Saving Processed Data
The final processed data (cleaned text and tokens) is saved as CSV files (one each for test and validation) for subsequent processing.

---

# 5. Fingerprinting Methods
This section describes the implementation and evaluation of the three fingerprinting methods.

## 5.1 MinHash
### 5.1.1 Hash Functions Generation
We generate a family of hash functions using the formula:
$$ h_{a,b}(x) = (a \times \text{mmh3.hash}(x) + b) \mod (2^{32} - 1) \tag{1} $$
where:
- $a$ and $b$ are random integers (different for each hash function)
- $\text{mmh3.hash}(x)$ is the MurmurHash3 function applied to input $x$
- We use a fixed random seed (`42`) for reproducibility

### 5.1.2 Signature Generation
For each document:
1. Convert the document into a set of shingles (word tokens or 3-word n-grams)
2. For each hash function in our family:
   - Apply the hash function to all shingles
   - Keep the minimum hash value as the signature component
3. The complete signature is the vector of these minimum values

**Parameters**:
- `num_perm` (128): Number of permutation functions (signature length)
- `num_bands` (32): Number of bands for LSH
- `rows_per_band` (4): Calculated as `num_perm/num_bands`
- `similarity_threshold` (0.7): Jaccard similarity threshold for considering documents duplicates

## 5.2 SimHash
### 5.2.1 Process
1. **Text to feature set conversion**:
   - Text is converted into a set of features tagged with its weight using TF-IDF.
2. **$f$-bit fingerprint generation**:
   - An $f$-dimensional vector $\vec{v}$ is initialized with each dimension set to zero.
   - Each feature is assigned a randomly generated $f$-bit vector (composed of -1 and 1). Multiply this vector by its corresponding weight and sum all vectors to get $\vec{v}$.
   - Assign 1 to the $i$-th bit if $\vec{v}_i > 0$; otherwise assign 0.

## 5.3 Bit Sampling
### 5.3.1 Principles and Implementation
Bit Sampling generates binary signatures where each bit represents the presence of specific content features:
- Signature length $n$ (default: 256 bits)
- Each token contributes to multiple bits based on term frequency (TF)

### 5.3.2 Signature Generation
For each document:
1. Tokenize text into word tokens (filtering tokens <3 characters)
2. Calculate term frequency (TF) for each token:
   $$ \text{tf}(t) = \frac{\text{count}(t)}{\text{total tokens}} \tag{2} $$
3. For each token $t$:
   - Generate a 32-bit hash $h(t)$
   - XOR this hash with $n$ pre-generated random seeds to create sub-hashes
   - Set signature bit to 1 if:
     $$ \text{sub\_hash} < \text{tf}(t) \times 2^{32} \tag{3} $$
4. Combine all tokens’ contributions via bitwise OR operations.

---

# 6. LSH Method
**Banding**: Divides hash signatures into smaller segments ("bands") and applies secondary hashing. Documents sharing identical band hashes become candidate pairs.

**MinHash Implementation**:
- 128-dimensional signatures divided into 32 bands (4 rows each)
- Parameters tuned for Jaccard similarity ≥0.7 detection

**SimHash Implementation**:
- 64-bit fingerprint with 4 bands (16 bits each)
- Band size adjusted to balance runtime and broadness

---

# 7. Similarity Confirmation Method
1. **MinHash Theoretical Basis**:
   - Jaccard similarity:
     $$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} \tag{4} $$
   - Threshold of 0.7 chosen empirically

2. **SimHash Hamming Distance**:
   - Similarity determined by XOR + bit count

---

# 8. Innovation
- **Union of Results**: Combine outputs from all three methods to reduce false negatives
- **Majority Voting**: Reduce false positives via consensus

---

# 9. Experimental Setup and Results
## 9.1 Results
- SimHash: Execution <10 minutes; high-accuracy similar pairs
- MinHash: 4,676 near-duplicates
- Bit Sampling: 1,402 near-duplicates
- SimHash: 1,201 near-duplicates

## 9.2 Running Time
| Method         | Time (seconds) | Breakdown                               |
|----------------|----------------|-----------------------------------------|
| MinHash        | 556            | Processing (188s), LSH (188s), Pairs (180s) |
| Bit Sampling   | 3347           | Tokenizing (27s), Signatures (220s), LSH (41s) |
| SimHash        | 220            | -                                       |

---

# 10. Method Comparison and Discussion
## 10.1 MinHash
- **Parameter Tuning**: 128 permutations, 32 bands optimized for Jaccard ≥0.7
- **Optimizations**: Vectorized operations, chunked processing, progress tracking

## 10.2 SimHash
- **Improvements**: Structural token removal, feature weighting adjustments
- **Parameter Tuning**: 8 bands balance runtime vs. pair detection

## 10.3 Bit Sampling
- **Performance**: Processed 325k documents at ~9.5k docs/sec
- **Configuration**: 256-bit signatures, 64 bands (4 bits each)

---

# 11. Limitations and Future Work
- **Limitations**: Parameter sensitivity, bucket size constraints, semantic understanding gaps
- **Future Work**: Adaptive parameter selection, hybrid semantic-lexical approaches

---

# 12. Conclusion
- **MinHash**: High recall (4,676 pairs)
- **SimHash**: High precision (1,201 pairs)
- **Bit Sampling**: Intermediate performance (1,402 pairs)

---

# 13. References
1. Lee et al. 2022. *Deduplicating Training Data Makes Language Models Better*. ACL.
2. Sadowski & Levin. 2011. *SimiHash: Hash-based Similarity Detection*. UCSC.
3. Charikar. 2002. *Similarity Estimation Techniques from Rounding Algorithms*. STOC.
4. Manku et al. 2007. *Detecting Near-Duplicates for Web Crawling*. WWW.
