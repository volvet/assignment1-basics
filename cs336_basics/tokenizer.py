from abc import ABC
from typing import List, Set, Iterable, Iterator, Tuple
from collections import defaultdict
import os
import regex
import regex as re
import numpy as np
import torch


GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = GPT2_TOKENIZER_REGEX

class Tokenizer(ABC):
    def encode(self, string: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

class BPETokenizer(Tokenizer):
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

    def get_bpe_merges(self, pieces: bytes) -> List[bytes]:
        parts = [bytes([b]) for b in pieces]
        while len(parts) > 1:
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.bpe_ranks:
                    pairs.add(pair)
            if not pairs:
                break

            best_pair = min(pairs, key = lambda pair: self.bpe_ranks[pair])
            #print('best pair:', best_pair)
            #print('parts before merge:', parts)
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
            #print('parts after merge:', parts)
            del pairs

        return parts


    def encode(self, string: str) -> List[int]:
        # Implement BPE encoding logic here
        if not string:
            return []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True) if self.special_tokens else []
        sorted_special_tokens_bytes = '|'.join(map(regex.escape, sorted_special_tokens))
        if self.special_tokens:
            chunks = regex.split(f'({sorted_special_tokens_bytes})', string)
        else:
            chunks = [string]
        #print('chunks:', chunks)
        final_ids = []
        for chunk in chunks:
            #print('processing chunk:', chunk)
            if not chunk:
                continue
            if chunk in self.special_tokens:
                #print("encoded special token:", chunk)
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue
            
            for word in regex.findall(GPT2_TOKENIZER_REGEX, chunk):
                if not word:
                    continue

                merged_pieces = self.get_bpe_merges(word.encode('utf-8'))
                #print('merged pieces:', merged_pieces)
                for piece in merged_pieces:
                    final_ids.append(self.bytes_to_id[piece])
            
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens: List[int]) -> str:
        # Implement BPE decoding logic here
        all_bytes = b''.join(self.vocab[id] for id in tokens)
        return all_bytes.decode("utf-8", errors="replace")


def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: List[str] = [],
              **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Implement BPE training logic here
    # print('special tokens:', special_tokens)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    if special_tokens:
        for token_bytes in special_token_bytes:
            if token_bytes not in vocab.values():
                vocab[next_id] = token_bytes
                next_id += 1

    if isinstance(input_path, os.PathLike):
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = input_path

    special_tokens_pattern = '|'.join(map(regex.escape, special_tokens)) if special_tokens else None
    if special_tokens_pattern:
        chunks = regex.split(f'({special_tokens_pattern})', text)
        #chunks = [chunk for chunk in chunks if chunk not in special_tokens]
    else:
        chunks = [text]

    #for i in range(min(5, len(chunks))):
    #    print('chunk', i, 'sample:', repr(chunks[i][:100]))
    pre_tokens_cnt = defaultdict(int)
    for chunk in chunks:
        #print('processing chunk:', chunk)
        if chunk in special_tokens:
            continue
        for m in regex.finditer(GPT2_TOKENIZER_REGEX, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1 

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        for token_tuple, cnt in pre_tokens_cnt.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair
        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1
        #print('best pair to merge:', best_pair, 'count:', pair_counts[best_pair])
        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    #print('Final vocab:', {k: v for k, v in vocab.items() if k >= 256})
    return vocab, merges

'''
def train_bpe2(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: List[str] = [],
              **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
# Step 1: Initialize Vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    # Step 2: Pre-tokenization
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')

    # Step 3: Compute BPE Merges
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all adjacent byte pairs
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    return vocab, merges
'''

if __name__ == "__main__":
    print("Training BPE tokenizer...")
    print("Using sample string 'the cat in the hat'")
    string = "the cat in the hat"  # @inspect string
    vocab, merges = train_bpe(string, vocab_size=256+3)  # @inspect params
    print("Vocab:")  # @inspect output
    for k, v in vocab.items():
        if k >= 256:
            print(f"{k}: {v}")

    print("Merges:")
    for merge in merges:
        print(merge)
