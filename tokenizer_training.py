from typing import List, Dict, Tuple, Optional
import regex as re
import pprint
import time
import sys
import json
import resource

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()


def pretokenize_text(text) -> List[str]:
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = re.findall(pretoken_regex, text)

    return pretokens


def init_vocab(special_tokens) -> (Dict[int, bytes], int):
    idx = 0
    vocabulary = {}
    for token in special_tokens:
        vocabulary[idx] = token.encode(ENCODING)
        idx += 1

    for i in range(256):
        vocabulary[idx] = i.to_bytes(1, 'big')
        idx += 1

    return vocabulary, idx


def string_to_bytes_list(input) -> List[bytes]:
    input_bytes = input.encode(ENCODING)
    input_byte_list = []

    for b in input_bytes:
        input_byte_list.append(bytes([b]))

    return input_byte_list


def digest_pretokens(pretokens) -> Dict[str, list]:
    pretoken_data = {}
    for token in pretokens:
        if token in pretoken_data:
            pretoken_data[token][0] += 1
        else:
            pretoken_data[token] = [1, string_to_bytes_list(token)]

    return pretoken_data


def count_pairs(pretoken_data) -> Dict[Tuple[bytes, bytes], int]:
    pair_counts = {}
    for pretoken in pretoken_data:
        pretoken_count = pretoken_data[pretoken][0]

        for first_tok, second_tok in zip(pretoken_data[pretoken][1][:-1], pretoken_data[pretoken][1][1:]):
            pair = (first_tok, second_tok)

            if pair in pair_counts:
                pair_counts[pair] += pretoken_count
            else:
                pair_counts[pair] = pretoken_count

    return pair_counts


def merge_step(pair_counts, vocabulary, idx, merges): 
    max_freq = max(list(pair_counts.values()))
    merge_pair = max([key for key in pair_counts if pair_counts[key] == max_freq])

    merged_token = merge_pair[0] + merge_pair[1]
    vocabulary[idx] = merged_token
    merges.append(merge_pair)
    idx += 1

    return idx


def map_pairs_to_pretokens(pretoken_data) -> (Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], List[str]]):
    pair_counts = {}
    pairs_to_pretokens = {}

    for pretoken in pretoken_data:
        pretoken_count = pretoken_data[pretoken][0]

        for first_tok, second_tok in zip(pretoken_data[pretoken][1][:-1], pretoken_data[pretoken][1][1:]):
            pair = (first_tok, second_tok)

            if pair in pair_counts:
                pair_counts[pair] += pretoken_count
            else:
                pair_counts[pair] = pretoken_count

            if pair in pairs_to_pretokens:
                if pretoken not in pairs_to_pretokens[pair]:
                    pairs_to_pretokens[pair].append(pretoken)
            else:
                pairs_to_pretokens[pair] = [pretoken]

    return pair_counts, pairs_to_pretokens


def check_nonneg_counts(count_dict) -> None:
    for key in count_dict:
        if count_dict[key] < 0:
            print('NEG COUNT!', key, count_dict[key])


def update_tokenization_of_pretoken(pretoken_tokens, merge_pair, merged_token, pretoken_count, pair_counts):
    new_pretoken_tokens = []
    i = 0
    considered_tokens = set()
    while i < len(pretoken_tokens):
        token1 = pretoken_tokens[i]

        if i < len(pretoken_tokens) - 1:
            token2 = pretoken_tokens[i + 1]
            token0 = None if i == 0 else pretoken_tokens[i - 1]
            token3 = None if i == len(pretoken_tokens) - 2 else pretoken_tokens[i + 2]

            if token1 + token2 == merged_token:
                pair_counts[merge_pair] -= pretoken_count

                new_pretoken_tokens.append(merged_token)

                # Reflect that the old pair has been broken
                if token0 and (i - 1, i) not in considered_tokens:
                    pair_counts[(token0, token1)] -= pretoken_count

                    if pair_counts[(token0, token1)] == 0:
                        del pair_counts[(token0, token1)]

                    considered_tokens.add((i - 1, i))

                if token3 and (i + 1, i + 2) not in considered_tokens:
                    pair_counts[(token2, token3)] -= pretoken_count

                    if pair_counts[(token2, token3)] == 0:
                        del pair_counts[(token2, token3)]

                    considered_tokens.add((i + 1, i + 2))

                i += 2
            else:
                new_pretoken_tokens.append(token1)
                i += 1
        else:
            new_pretoken_tokens.append(token1)
            i += 1

    return new_pretoken_tokens#, pair_counts


def redo_pair_counts_after_merge(pretoken_data, pretoken, merged_token, pair_counts, pairs_to_pretokens, pretoken_count): 
    for first_tok, second_tok in zip(pretoken_data[pretoken][1][:-1], pretoken_data[pretoken][1][1:]):
        pair = (first_tok, second_tok)

        if merged_token in pair:
            if pair in pair_counts:
                pair_counts[pair] += pretoken_count
            else:
                pair_counts[pair] = pretoken_count

        if pair in pairs_to_pretokens:
            if pretoken not in pairs_to_pretokens[pair]:
                pairs_to_pretokens[pair].append(pretoken)
        else:
            pairs_to_pretokens[pair] = [pretoken]



def update_pretoken_data_fast(pretoken_data, pairs_to_pretokens, merge_pair, pair_counts):
    # Look at only the pretokens involving the merged pair
    for pretoken in pairs_to_pretokens[merge_pair]:
        pretoken_count = pretoken_data[pretoken][0]
        merged_token = merge_pair[0] + merge_pair[1]
        pretoken_tokens = pretoken_data[pretoken][1]

        # Update tokenization of the pretoken
        new_pretoken_tokens = update_tokenization_of_pretoken(pretoken_tokens=pretoken_tokens, merge_pair=merge_pair, merged_token=merged_token, pretoken_count=pretoken_count, pair_counts=pair_counts)

        pretoken_data[pretoken][1] = new_pretoken_tokens

        # Redo pair counts based on new tokenization
        redo_pair_counts_after_merge(pretoken_data=pretoken_data, pretoken=pretoken, merged_token=merged_token, pair_counts=pair_counts, pairs_to_pretokens=pairs_to_pretokens, pretoken_count=pretoken_count)


def vocab_formatting(vocab, special_tokens):
    new_vocab = {}
    for key in vocab:
        if key >= 256 + len(special_tokens):
            new_vocab[key] = vocab[key].decode(encoding=ENCODING, errors='replace')

    return new_vocab


def merges_formatting(merges):
    new_merges = []
    for elem in merges:
        new_merges.append([elem[0].decode(encoding=ENCODING, errors='replace'), elem[1].decode(encoding=ENCODING, errors='replace')])

    return new_merges


def train_tokenizer(input_path: str,
                    vocab_size: int,
                    special_tokens: List[str]) -> (Dict[int, bytes], List[Tuple[bytes, bytes]]):
    overall_start = time.time()
    # Define key data var
    merges = []

    # Initialize the vocabulary with special tokens (byte encodings) and single byte values
    vocabulary, idx = init_vocab(special_tokens=special_tokens)

    # Pretokenize text and digest pretoken data into dict of {str: [count, tokenized str in list form]}
    pretoken_data = digest_pretokens(pretokens=pretokenize_text(text=open(input_path).read()))

    # Create mapping of pairs to pretokens they're in
    pair_counts, pairs_to_pretokens = map_pairs_to_pretokens(pretoken_data=pretoken_data)

    # Run BPE
    while len(vocabulary) < vocab_size:
        # Merge the most frequent pair and add it to the vocab
        idx = merge_step(pair_counts=pair_counts, vocabulary=vocabulary, merges=merges, idx=idx)

        # Update pretoken data and counts
        update_pretoken_data_fast(pretoken_data=pretoken_data, pairs_to_pretokens=pairs_to_pretokens, merge_pair=merges[-1], pair_counts=pair_counts)

    overall_end = time.time()

    with open('trained_vocab_owt.json', 'w') as f:
        json.dump(vocab_formatting(vocabulary, special_tokens), f, indent=4)

    with open('merges_owt.json', 'w') as f:
        f.write(json.dumps(merges_formatting(merges), indent=4))

    print('| total time:', overall_end - overall_start)

    return vocabulary, merges


def main():
    args = sys.argv
    if len(args) > 1:
        fp = args[1]
        vocab_size = int(args[2])
        if len(args) > 3:
            special_tokens = args[3]
            special_tokens = special_tokens.split(',')
        else:
            special_tokens = None
    else:
        fp = 'cs336_basics/speed_test.txt'
        vocab_size = 500
        special_tokens = ['<|endoftext|>']

    print('ARGS! fp:', fp, ', vocab size:', vocab_size, ', special tokens:', special_tokens)

    vocab, merges = train_tokenizer(input_path=fp, vocab_size=vocab_size, special_tokens=special_tokens)


if __name__ == '__main__':
    main()
