from abc import ABC
from typing import List, Set, Iterable, Iterator
import regex
import numpy as np
import torch


GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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