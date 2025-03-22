from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizerSimple:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab  = {}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special = {'<|EOS|>'}):
        procesed_text = []
        for i, char in enumerate(text):
            if char == ' ' and i !=0:
                procesed_text.append('Ġ')  #Ġ -> paricularity of gpt-2 bpe implementation
            if char != ' ':
                procesed_text.append(char)
        procesed_text = ''.join(procesed_text)

        # starts with first 256 ASCII chars
        unique_chars = [chr(i) for i in range(256)]

        unique_chars.extend(char for char in sorted(set(procesed_text)) if char not in unique_chars)

        if 'Ġ' not in unique_chars:
            unique_chars.extend('Ġ')


        # create voacb and inv_vocab
        self.vocab = {i:char for i, char in enumerate(unique_chars)}
        self.inv_vocab = {char: i for i, char in self.vocab.items()}

        # add special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inv_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inv_vocab[token] = new_id

        # tokenizeh the processed_text into token_ids
        token_ids = [self.inv_vocab[char] for char in procesed_text]

        # BPE algorithm (find frequent pairs and replace)
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode= 'most')
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # bulid the vocab
        # for (p0, p1), new_id in self.bpe_merges.items():