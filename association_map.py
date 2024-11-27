
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from queue import Queue

class AssociationMap:

    def __init__(self, tokens):

        self.token_set = set([token.lower().strip() for token in tokens])
        token_count = len(self.token_set)
        self.token_list = sorted(list(self.token_set))

        # create a dataframe with tokens as indices and column headers
        raw_token_map_array = np.zeros(shape=(token_count, token_count))
        self.raw_token_map = pd.DataFrame(raw_token_map_array, columns=self.token_list, index=self.token_list)
        self.token_map = None # this will be the standardized version, after calling standardize_token_map

        # create two dataframes mapping token couplets to tokens
        # one dataframe will hold the counts, one the sums, average can be computed using both
        raw_couplet_map_array = np.zeros(shape=(int(((token_count*(token_count-1))/2)), token_count))
        index = []
        for token1 in self.token_list:
            for token2 in self.token_list:
                if token1 < token2:
                    index.append((token1, token2))
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

    # this should be run after add_text, or after add_text*(n times)
    def standardize_token_map(self):
        print('0.00%', end='\r')
        count = 0
        self.token_map = self.raw_token_map.div(self.raw_token_map.sum(axis=1), axis=0)
        self.token_map = self.token_map.fillna(0)
        doubled_tokens = set()
        for token1 in self.token_list:
            for token2 in self.token_list:
                if token1 != token2 or token1 not in doubled_tokens:
                    token1_score = self.token_map.at[token1, token2]
                    token2_score = self.token_map.at[token2, token1]
                    score = token1_score * token2_score
                    self.token_map.at[token1, token2] = score
                    if token1 == token2:
                        doubled_tokens.add(token1)
            count += 1
            if count % 50 == 0:
                print(f'{round((count*100)/len(self.token_list), 2)}%', end='\r')
        print('100.00%')

    #get the associated row, whether index is token or token couplet
    def get_association_row(self, token, normalize):
        if isinstance(token, str):
            row = self.token_map.loc[token, :].copy()
        elif isinstance(token, tuple):
            row_counts = self.raw_couplet_map_counts.loc[[token]].copy()
            row_sums = self.raw_couplet_map_sums.loc[[token]].copy()
            row = row_sums / row_counts
            row = row.fillna(0)
        if normalize:
            row_sum = row.sum()
            if row_sum == 0:
                row_sum = 1
            row /= row_sum
        row_as_tuple = tuple(zip(tuple(self.token_list), tuple(row.values.flatten())))
        return row_as_tuple


def get_related_tokens(association_map, input_string, max_depth):
    
    related_tokens = {}
    input_tokens = word_tokenize(input_string.lower())

    branch_factor = 8
    global_level = 0

    token_queue = Queue()
    for input_token in input_tokens:
        token_queue.put((input_token, input_token, 0, 1))
        for input_token_alt in input_tokens:
            if input_token < input_token_alt:
                token_queue.put(((input_token, input_token_alt), (input_token, input_token_alt), 0, 1))
   
    while not token_queue.empty():

        key_quartet = token_queue.get()
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

        if isinstance(key, str):
            entry = related_tokens.get(key, [])
            entry.append((origin_token, focus))
            related_tokens[key] = entry

        association_tuple = association_map.get_association_row(key, normalize=False)
        top_associated_tokens = sorted(association_tuple, key=lambda x: x[1], reverse=True)[:branch_factor]
        for associated_token in top_associated_tokens:
            associated_token_score = associated_token[1] * focus
            token_quartet = (associated_token[0], origin_token, local_level+1, associated_token_score)
            token_queue.put(token_quartet)

    related_tokens_merged = {}
    # for related_tokens, destination tokens are keys, list of origin token couplets are values
    # the list of origin tokens and values is necessary above, but the merged sum is relevant for output
    for destination_token in related_tokens:
        origin_couplets = related_tokens[destination_token]
        destination_sum = 0
        origins = set()
        for origin_couplet in origin_couplets:
            # for next paragraph, only works with string values
            if isinstance(origin_couplet[0], str):
                origins.add(origin_couplet[0])
            destination_sum += origin_couplet[1]
        related_tokens_merged[destination_token] = destination_sum

        # update the association_map for future uses of this function
        origin_pairs_sum = destination_sum / len(origins)
        for origin1 in origins:
            for origin2 in origins:
                if origin1 < origin2:
                    couplet = (origin1, origin2)
                    association_map.raw_couplet_map_counts.at[couplet, associated_token[0]] += 1
                    association_map.raw_couplet_map_sums.at[couplet, associated_token[0]] += origin_pairs_sum

    return related_tokens_merged
            
            