def sample_from_related_tokens(self, token, size):
        token_row = self.token_map.loc[token, :]
        token_row_sum = token_row.sum()
        if token_row_sum == 0:
            token_row_sum = 1
        probabilities = token_row / token_row.sum()
        sampled_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
        sampled_tokens = [(self.token_list[i], probabilities[i]) for i in sampled_indices]
        return sampled_tokens
    
    def update_related_tokens(self, token, related_tokens, focus, branching):
        if branching <= 1:
            return
        sampled_tokens = self.sample_from_related_tokens(token, branching)
        print(token, sampled_tokens, focus)
        for sampled_token in sampled_tokens:
            focus = focus*sampled_token[1]
            related_tokens[sampled_token[0]] = related_tokens.get(sampled_token[0], 0) + focus
            self.update_related_tokens(sampled_token, related_tokens, focus, branching//2)

    def get_related_words(self, input_string):
        input_tokens = word_tokenize(input_string.lower())
        related_tokens_meta_dict = {}
        for input_token in input_tokens:
            related_tokens = {}
            self.update_related_tokens(input_token, related_tokens, 1, 8)
            related_tokens_meta_dict[input_token] = related_tokens
        return related_tokens_meta_dict