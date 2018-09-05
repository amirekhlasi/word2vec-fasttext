# coding: utf-8

import make_indexed_corpus as mic
import numpy as np
import pickle

def word_embedding(word, subvocab, sub_W1, sub_W2, min_subword_length, max_subword_length):
    sub_words = [subvocab[sw] for sw in mic.range_subwords(word, min_subword_length, max_subword_length)]
    W1 = sub_W1[sub_words].sum(0)
    W2 = sub_W2[sub_words].sum(0)
    return W1, W2

def dict_word_embedding(vocab, subvocab, sub_W1, sub_W2, subvocab_length):
    W1 = []
    W2 = []
    for word in vocab:
        w1, w2 = word_embedding(word, subvocab, sub_W1, sub_W2, subvocab_length)
        W1.append(w1)
        W2.append(w2)
    return np.array(W1), np.array(W2)



if __name__ == "__main__":
    with open("Data/vocab","rb") as pk:
        vocab = pickle.load(pk)
    with open("Data/subvocab", "rb") as pk:
        subvocab_length, subvocab = pickle.load(pk)
    sub_W1, sub_W2 = np.load("sub_embedding.npy")
    W = np.array(dict_word_embedding(vocab, subvocab, sub_W1, sub_W2, subvocab_length))
    np.save("word_embedding.npy", W)
    np.savetxt("word_embedding_1.txt", W[0])
    np.savetxt("word_embedding_2.txt", W[1])
