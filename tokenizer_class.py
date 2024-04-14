from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Iterable, Iterator
import regex as re
import pprint
import time
import sys
import json

import json
import os
import resource
import sys
from typing import Optional

import pytest
import tiktoken

from itertools import chain

import resource
import tracemalloc

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes] = None, merges: List[Tuple[bytes, bytes]] = None, special_tokens: List[str] = None):
        self.id_to_tok = vocab
        if vocab:
            self.tok_to_id = {val: key for key, val in self.id_to_tok.items()}
        else:
            self.tok_to_id = None
        self.merges = merges
        self.special_tokens = special_tokens

    def init_vocab(self, special_tokens) -> Dict[int, bytes]:
        idx = 0
        vocabulary = {}
        if special_tokens:
            for token in special_tokens:
                vocabulary[idx] = token.encode(ENCODING)
                idx += 1

        for i in range(256):
            vocabulary[idx] = i.to_bytes(1, 'big')
            idx += 1

        return vocabulary

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = self.init_vocab(special_tokens=special_tokens)

        with open(vocab_filepath, 'r') as f:
            vocab_added = json.load(f)

        vocab_added = {int(key): val.encode(ENCODING) for key, val in vocab_added.items()}
        vocab.update(vocab_added)

        with open(merges_filepath, 'r') as f:
            merges = json.load(f)

        merges = [(elem[0].encode(ENCODING), elem[1].encode(ENCODING)) for elem in merges]

        new_tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

        return new_tokenizer

    def pretokenize_text(self, text: str):
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len)  # Short to long
        else:
            special_tokens = []
        normal_pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        def split_text(block: str, remaining_tokens: List[str]):
            if len(remaining_tokens) == 0:
                return re.findall(normal_pretoken_regex, block)
            else:
                current_token = remaining_tokens[-1]
                sub_blocks = block.split(current_token)
                final_tokens = []

                for sub_block in sub_blocks:
                    if len(sub_block) > 0:
                        final_tokens += split_text(sub_block, remaining_tokens[:-1])
                    final_tokens += [current_token]

                return final_tokens[:-1]

        all_final_tokens = split_text(text, special_tokens)

        return all_final_tokens

    def string_to_bytes_list(self, input) -> List[bytes]:
        input_bytes = input.encode(ENCODING)
        input_byte_list = []

        for b in input_bytes:
            input_byte_list.append(bytes([b]))

        return input_byte_list

    def merge_tokens(self, pretokens: List[str]) -> List[bytes]:
        final_tokens = []
        counter = 0
        for pretoken in pretokens:
            if self.special_tokens and pretoken in self.special_tokens:
                final_tokens.append(pretoken.encode(ENCODING))
                continue

            pretoken_bytes = self.string_to_bytes_list(pretoken)
            old_pretoken_bytes = pretoken_bytes
            new_pretoken_bytes = []

            for merge in self.merges:
                i = 0
                while i < len(old_pretoken_bytes):
                    token1 = old_pretoken_bytes[i]

                    if i < len(old_pretoken_bytes) - 1:
                        token2 = old_pretoken_bytes[i + 1]

                        if (token1, token2) == merge:
                            new_pretoken_bytes.append(token1 + token2)
                            i += 1
                        else:
                            new_pretoken_bytes.append(token1)
                    else:
                        new_pretoken_bytes.append(token1)

                    i += 1

                old_pretoken_bytes = new_pretoken_bytes
                new_pretoken_bytes = []

            final_tokens += old_pretoken_bytes
            counter += 1
            if counter % 1000 == 0:
                print('merge done', counter)

        return final_tokens

    def map_tokens_to_ids(self, tokens: List[bytes]) -> List[int]:
        ids = []

        for token in tokens:
            ids.append(self.tok_to_id[token])

        return ids

    def encode(self, text: str) -> List[int]:
        # print('input to encode', text)

        # First, pretokenize text
        pretokens = self.pretokenize_text(text=text)

        # Second, merge bytes within pretokens according to tokenizer's sequence of merges
        final_tokens = self.merge_tokens(pretokens=pretokens)

        # Finally, map bytes to token ids
        token_ids = self.map_tokens_to_ids(tokens=final_tokens)

        return token_ids

    def encode_iterable_old(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(text=line)

    # # # # # # Core lazy loading/iterable functions below
    def pretokenize_text_lazy(self, text: str):
        normal_pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        def split_text(block: str, remaining_tokens: List[str]):
            if len(remaining_tokens) == 0:
                yield from re.finditer(normal_pretoken_regex, block)
            else:
                current_token = remaining_tokens[-1]
                sub_blocks = block.split(current_token)     # TODO: this loads the whole line?

                count = 0
                total_len = len(sub_blocks)

                for sub_block in sub_blocks:
                    if len(sub_block) > 0:
                        if count < total_len - 1:
                            yield from chain((elem[0] for elem in split_text(sub_block, remaining_tokens[:-1])), (elem for elem in [current_token]))
                        else:
                            yield from (elem[0] for elem in split_text(sub_block, remaining_tokens[:-1]))
                    elif count < total_len - 1:
                        yield from [current_token]

                    count += 1

        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len)  # Short to long
            all_final_tokens = split_text(text, special_tokens)
        else:
            all_final_tokens = (elem[0] for elem in re.finditer(normal_pretoken_regex, text))

        yield from all_final_tokens

    def merge_tokens_lazy(self, pretokens):
        for pretoken in pretokens:
            if self.special_tokens and pretoken in self.special_tokens:
                yield [pretoken.encode(ENCODING)]
                continue

            pretoken_bytes = self.string_to_bytes_list(pretoken)
            old_pretoken_bytes = pretoken_bytes
            new_pretoken_bytes = []

            for merge in self.merges:
                i = 0
                while i < len(old_pretoken_bytes):
                    token1 = old_pretoken_bytes[i]

                    if i < len(old_pretoken_bytes) - 1:
                        token2 = old_pretoken_bytes[i + 1]

                        if (token1, token2) == merge:
                            new_pretoken_bytes.append(token1 + token2)
                            i += 1
                        else:
                            new_pretoken_bytes.append(token1)
                    else:
                        new_pretoken_bytes.append(token1)

                    i += 1

                old_pretoken_bytes = new_pretoken_bytes
                new_pretoken_bytes = []

            yield old_pretoken_bytes

    def map_tokens_to_ids_lazy(self, tokens):
        for token in tokens:
            for sub_token in token:
                yield self.tok_to_id[sub_token]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for elem in iterable:
            pretokens = self.pretokenize_text_lazy(text=elem)

            final_tokens = self.merge_tokens_lazy(pretokens=pretokens)

            token_ids = self.map_tokens_to_ids_lazy(tokens=final_tokens)

            yield from token_ids
    # # # # # # # # Core lazy loading/iterable functions above
  
    def decode(self, ids: List[int]) -> str:
        all_bytes = b''
        for b in [self.id_to_tok[id] for id in ids]:
            all_bytes += b

        decoded_str = all_bytes.decode(encoding=ENCODING, errors='replace')

        # print('decoded str from ids:', decoded_str)

        return decoded_str


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Test functions, without decorators
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given the path to a JSON vocab, a file with BPE merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: Optional[list[str]] = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)


def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


def _encode(tokenizer, text):
    """
    We place tokenizer.encode into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    return tokenizer.encode(text)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def main():
    tokenizer = Tokenizer()


if __name__ == '__main__':
    main()
