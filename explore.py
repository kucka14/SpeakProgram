import importlib
import association_map as am
import numpy as np

if __name__ == '__main__':

    with open('texts/1000-most-common-words.txt', 'r') as f:
        tokens = f.readlines()
    with open('texts/alice_in_wonderland.txt', 'r') as f:
        alice_text = f.read()
    with open('texts/count_of_monte_cristo.txt', 'r') as f:
        monte_cristo_text = f.read()

    association_map = am.AssociationMap(tokens)

    association_map.add_text(alice_text)
    association_map.add_text(monte_cristo_text)

    association_map.standardize_token_map()

    # see if the output changes after multiple calls; couplet map should be changing/converging
    for i in range(5):
        related_tokens_dict = am.get_related_tokens(association_map, 'glass water', 4)
        sorted_related_tokens = sorted(list(related_tokens_dict.items()), key=lambda x: x[1], reverse=True)
        print(sorted_related_tokens)
        print()