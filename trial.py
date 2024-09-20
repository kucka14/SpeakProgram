import importlib
import association_map as am
import numpy as np

if __name__ == '__main__':

    with open('1000-most-common-words.txt', 'r') as f:
        tokens = f.readlines()
    with open('alice_in_wonderland.txt', 'r') as f:
        alice_text = f.read()
    with open('count_of_monte_cristo.txt', 'r') as f:
        monte_cristo_text = f.read()

    association_map = am.AssociationMap(tokens)

    association_map.add_text(alice_text)

    association_map.standardize_token_map()

    print(am.get_related_tokens(association_map, 'glass water', 4))