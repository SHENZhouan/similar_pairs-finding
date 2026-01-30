import gzip, math, os, random, re, sys, hashlib
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import xxhash as xxh
    xxhash_avail = True
except ImportError:
    xxhash_avail = False


class BitSamplingLSH:
    """
    仅用 Bit‑Sampling 生成 0/1 位签名，再用 LSH(band/hash) 选候选对，
    不使用 MinHash / SimHash。
    """
    WORD_RE = re.compile(r"\b\w+\b")

    def __init__(
        self,
        signature_len: int = 256,
        num_bands: int = 64,
        band_width: int = None,           # 若 None 则自动 = ceil(signature_len/num_bands)
        content_weight: float = 0.7,
        sim_threshold: float = 0.7,
        min_content_sim: float = 0.6,
        max_bucket_size: int = 50_000,
        seed: int = 42,
    ):
        self.sig_len = signature_len
        self.num_bands = num_bands
        self.band_width = (
            band_width if band_width else math.ceil(signature_len / num_bands)
        )
        self.content_weight = content_weight
        self.sim_threshold = sim_threshold
        self.min_content_sim = min_content_sim
        self.max_bucket = max_bucket_size

        np.random.seed(seed)
        # 32 位无符号随机种子，用于 XOR
        self.seeds = np.random.randint(0, 2 ** 31 - 1, size=self.sig_len, dtype="uint32")

        print(
            f"BitSamplingLSH ready: sig={self.sig_len}, bands={self.num_bands}, "
            f"band_width={self.band_width}, weight={self.content_weight}"
        )

    # ---------- 文本预处理 ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        toks = BitSamplingLSH.WORD_RE.findall(text.lower())
        # 去掉过短 token
        return [t for t in toks if len(t) > 2]

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        cnt = {}
        for t in tokens:
            cnt[t] = cnt.get(t, 0) + 1
        N = float(len(tokens))
        return {t: c / N for t, c in cnt.items()}

    # ---------- 生成 Bit‑Sampling 签名 ----------
    def _fast_hash32(self, token: str) -> int:
        """尽量用 xxhash, 否则 fallback 到 builtin hash."""
        if xxhash_avail:
            return xxh.xxh32(token).intdigest()
        return hash(token) & 0xFFFFFFFF

    def build_signature(self, tokens: List[str]) -> np.ndarray:
        """
        向量化实现：
        对每个 token 先得到一个 32 位整数 h，
        然后与 seeds XOR 得到 length 个子 hash，比较阈值决定置 1。
        最后对所有 token 做 OR。时间复杂度  O(|T| * sig_len) 但在 C 端完成。
        """
        if not tokens:
            return np.zeros(self.sig_len, dtype=np.bool_)

        tf = self._tf(tokens)
        sig = np.zeros(self.sig_len, dtype=np.bool_)

        seeds = self.seeds  # uint32[signature_len]

        for token, weight in tf.items():
            h = self._fast_hash32(token)
            # 与 seeds 向量化 XOR，再与阈值比较
            xor_res = (h ^ seeds).astype("uint32")
            thresh = np.uint32(weight * 0xFFFFFFFF)
            sig |= xor_res < thresh
            # 小技巧：numpy 会把 bool|= 运算自动 vectorize

            # 早退：如果已经全 1 了再继续也不会改变
            if sig.all():
                break
        return sig

    # ---------- LSH ----------
    def _band_hash(self, band_bits: np.ndarray) -> int:
        """
        把一段 bool 向量 pack 成 bytes，然后 Python 内置 hash(bytes)，
        速度比 tuple 快很多。
        """
        packed = np.packbits(band_bits.astype(np.uint8))
        return hash(packed.tobytes())

    def build_lsh_index(
        self, sigs: Dict[int, np.ndarray]
    ) -> Dict[str, List[int]]:
        buckets = defaultdict(list)
        for doc_id, sig in tqdm(sigs.items(), desc="LSH‑index"):
            bands_indexed = 0
            for b in range(self.num_bands):
                s = b * self.band_width
                e = min(s + self.band_width, self.sig_len)
                if s >= self.sig_len:
                    break
                band = sig[s:e]

                if not band.any() or band.all():
                    continue  # 太不区分

                key = f"{b}_{self._band_hash(band)}"
                if len(buckets[key]) < self.max_bucket:
                    buckets[key].append(doc_id)
                    bands_indexed += 1

            if bands_indexed < 3:  # 兜底随机桶
                rnd_key = f"bk_{np.sum(sig)}"
                buckets[rnd_key].append(doc_id)

        # 去掉只有 1 篇 or 超大桶（>15000）
        out = {k: v for k, v in buckets.items() if 1 < len(v) <= 15000}
        print(
            f"index buckets: total {len(buckets)}, keep {len(out)}, "
            f"drop {len(buckets) - len(out)}"
        )
        return out

    # ---------- 相似度 ----------
    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _similarity(
        self,
        tok1: List[str],
        tok2: List[str],
        sig1: np.ndarray,
        sig2: np.ndarray,
    ) -> Tuple[float, float, float]:
        bit_sim = np.mean(sig1 == sig2)
        if bit_sim < 0.5:
            return bit_sim, 0.0, 0.0
        cont_sim = self._jaccard(tok1, tok2)
        if cont_sim < self.min_content_sim:
            return bit_sim, cont_sim, 0.0
        comb = (1 - self.content_weight) * bit_sim + self.content_weight * cont_sim
        return bit_sim, cont_sim, comb

    # ---------- 主流程 ----------
    def detect(
        self,
        documents: Dict[int, str],
        doc_sources: Dict[int, str] = None,
    ) -> List[Dict]:
        print("tokenising ...")
        doc_tokens = {
            i: self._tokenize(txt) for i, txt in tqdm(documents.items())
        }
        # 去空
        doc_tokens = {k: v for k, v in doc_tokens.items() if v}
        print(f"valid docs: {len(doc_tokens)}")

        print("signatures ...")
        signatures = {
            i: self.build_signature(tok) for i, tok in tqdm(doc_tokens.items())
        }

        # 多样性 quick check
        sample = list(signatures.values())[:1000]
        uniq = {tuple(sig[:32]) for sig in sample}
        if len(uniq) < len(sample) * 0.1:
            print("⚠️  低签名多样性, 可能假阳性过多")

        buckets = self.build_lsh_index(signatures)

        # -------- 生成候选对 --------
        print("generate candidate pairs ...")
        cand_pairs = set()
        for docs in tqdm(buckets.values()):
            if len(docs) > 500:  # big bucket，随机采样
                docs = random.sample(docs, 500)
            # itertools.combinations 速度快 & 不生成重复
            cand_pairs.update(
                (min(a, b), max(a, b)) for a, b in combinations(docs, 2)
            )
        print(f"candidates: {len(cand_pairs):,}")

        # -------- 评估相似度 --------
        results = []
        for d1, d2 in tqdm(cand_pairs, desc="similarity"):
            bit_sim, cont_sim, comb = self._similarity(
                doc_tokens[d1], doc_tokens[d2], signatures[d1], signatures[d2]
            )
            if comb >= self.sim_threshold:
                info = dict(
                    doc_id_1=d1,
                    doc_id_2=d2,
                    bit_similarity=bit_sim,
                    content_similarity=cont_sim,
                    combined_similarity=comb,
                    preview_1=" ".join(doc_tokens[d1][:12]),
                    preview_2=" ".join(doc_tokens[d2][:12]),
                )
                if doc_sources:
                    info["source_1"] = doc_sources.get(d1, "unknown")
                    info["source_2"] = doc_sources.get(d2, "unknown")
                results.append(info)

        print(
            f"similar docs ≥ {self.sim_threshold}: {len(results):,} "
            f"(checked {len(cand_pairs):,} pairs)"
        )
        return results


# ---------------- 辅助函数 ----------------
def load_documents(
    file_paths: Dict[str, str], columns: Dict[str, str] = None
) -> Tuple[Dict[int, str], Dict[int, str]]:
    docs, src = {}, {}
    idx = 0
    for name, path in file_paths.items():
        print(f"loading {name} ...")
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            col = (columns or {}).get("content", "content")
            for text in df[col].astype(str):
                docs[idx] = text
                src[idx] = name
                idx += 1
        elif path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    # 假设  tsv  第二列是内容
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        docs[idx] = parts[1]
                        src[idx] = name
                        idx += 1
    print(f"loaded {len(docs)} docs")
    return docs, src


def main():
    in_files = {
        "test": "cleaned_preprocessed_test.csv",
        "valid": "cleaned_preprocessed_validation.csv",
    }
    docs, src = load_documents(in_files, columns={"content": "cleaned_text"})

    detector = BitSamplingLSH(
        signature_len=256,
        num_bands=64,        # 这里可自由改；band_width 会自动调整
        content_weight=0.7,
        sim_threshold=0.7,
        min_content_sim=0.6,
    )

    pairs = detector.detect(docs, src)

    out_csv = "bit_sampling_similar_pairs_try2.csv"
    pd.DataFrame(pairs).to_csv(out_csv, index=False)
    print("done ->", out_csv)


if __name__ == "__main__":
    main()
