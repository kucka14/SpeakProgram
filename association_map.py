
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import math

class AssociationMap:

    def __init__(self, tokens):

        self.token_set = set([token.lower().strip() for token in tokens])
        token_count = len(self.token_set)
        self.token_list = sorted(list(self.token_set))
        raw_token_map_array = np.zeros(shape=(token_count, token_count))
        self.raw_token_map = pd.DataFrame(raw_token_map_array, columns=self.token_list, index=self.token_list)
        self.token_map = None

        raw_couplet_map_array = np.zeros(shape=(int(((token_count*(token_count-1))/2)+token_count), token_count))
        index = []
        doubled_tokens = set()
        for token1 in self.token_list:
            for token2 in self.token_list:
                if token1 < token2:
                    index.append((token1, token2))
                elif token1 == token2 and token1 not in doubled_tokens:
                    index.append((token1, token2))
                    doubled_tokens.add(token1)
        self.raw_couplet_map_counts = pd.DataFrame(raw_couplet_map_array, columns=self.token_list, index=index)
        self.raw_couplet_map_sums = pd.DataFrame(raw_couplet_map_array, columns=self.token_list, index=index)

    def add_text(self, text):
        print('0.00%', end='\r')
        text = text.lower()
        text_tokens = word_tokenize(text)
        pointer = 0
        while pointer < len(text_tokens) - 1:
            current_token = text_tokens[pointer]
            if current_token in self.token_set:
                weight = 10
                subpointer = pointer + 1
                while weight > 0 and subpointer < len(text_tokens):
                    next_token = text_tokens[subpointer]
                    if next_token in self.token_set:
                        self.raw_token_map.at[current_token, next_token] += weight
                        self.raw_token_map.at[next_token, current_token] += weight
                    weight -= 1
                    subpointer += 1
            pointer += 1
            if pointer % 500 == 0:
                print(f'{round((pointer*100)/len(text_tokens), 2)}%', end='\r')
        print('100.00%')

    def standardize_token_map(self):
        print('0.00%', end='\r')
        count = 0
        self.token_map = self.raw_token_map.div(self.raw_token_map.sum(axis=1), axis=0)
        self.token_map = self.token_map.fillna(0)
        doubled_words = set()
        for word1 in self.token_list:
            for word2 in self.token_list:
                if word1 != word2 or word1 not in doubled_words:
                    word1_score = self.token_map.at[word1, word2]
                    word2_score = self.token_map.at[word2, word1]
                    score = word1_score * word2_score
                    self.token_map.at[word1, word2] = score
                    if word1 == word2:
                        doubled_words.add(word1)
            count += 1
            if count % 50 == 0:
                print(f'{round((count*100)/len(self.token_list), 2)}%', end='\r')
        print('100.00%')

    def get_association_row(self, token, normalize):
        if isinstance(token, str):
            row = self.token_map.loc[token, :].copy()
        elif isinstance(token, tuple):
            row_counts = self.raw_couplet_map_count.loc[[token]].copy()
            row_sums = self.raw_couplet_map_sums.loc[[token]].copy()
            row = row_sums / row_counts
        if normalize:
            row_sum = row.sum()
            if row_sum == 0:
                row_sum = 1
            row /= row_sum
        return tuple(zip(tuple(self.token_list), tuple(row)))
            

class TokenQueue:

    def __init__(self):
        self.queue0 = []
        self.queue1 = []
        self.queue2 = []
        self.queue3 = []

    def put(self, item):
        if isinstance(item[0], str) and item[0] in self.queue0:
            item_index = self.queue0.index(item[0])
            origin_token1 = self.queue1[item_index]
            origin_token2  = item[1]
            self.queue1[item_index] = item[1]
            self.queue2[item_index] = max(self.queue2[item_index], item[2])
            self.queue3[item_index] += item[3]
            return (origin_token1, origin_token2, self.queue3[item_index])
        else:
            self.queue0.append(item[0])
            self.queue1.append(item[1])
            self.queue2.append(item[2])
            self.queue3.append(item[3])
            return None

    def get(self):
        if len(self.queue0) == 0:
            return None
        return (self.queue0.pop(0), self.queue1.pop(0), self.queue2.pop(0), self.queue3.pop(0))
    
    def is_empty(self):
        return len(self.queue0) == 0


def get_related_tokens(association_map, input_string, max_depth):

    related_tokens = {}
    input_tokens = word_tokenize(input_string.lower())

    branch_factor = 8
    global_level = 0

    tq = TokenQueue()
    for input_token in input_tokens:
        tq.put((input_token, input_token, 0, 1))
        for input_token_alt in input_tokens:
            if input_token < input_token_alt:
                tq.put(((input_token, input_token_alt), (input_token, input_token_alt), 0, 1))
    while not tq.is_empty():

        key_quartet = tq.get()
        key = key_quartet[0]
        origin_token = key_quartet[1]
        local_level = key_quartet[2]
        focus = key_quartet[3]

        if local_level != global_level:
            if local_level == max_depth:
                break
            if local_level == 1:
                branch_factor = 2
            global_level = local_level

        related_tokens[key] = related_tokens.get(key, 0) + focus

        normalize = False
        if isinstance(key, str):
            normalize = True
        association_tuple = association_map.get_association_row(key, normalize)
        top_associated_tokens = sorted(association_tuple, key=lambda x: x[1], reverse=True)[:branch_factor]
        for associated_token in top_associated_tokens:
            associated_token_score = associated_token[1] * focus
            token_quartet = (associated_token[0], origin_token, local_level+1, associated_token_score)
            couplet_map_update = tq.put(token_quartet)
            if couplet_map_update is not None:
                token1 = couplet_map_update[0]
                token2 = couplet_map_update[1]
                if token1 <= token2:
                    couplet = (token1, token2)
                else:
                    couplet = (token2, token1)
                association_map.raw_couplet_map_counts.at[couplet, associated_token[0]] += 1
                association_map.raw_couplet_map_sums.at[couplet, associated_token[0]] += associated_token_score

    return related_tokens
            
            