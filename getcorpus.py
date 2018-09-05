# coding: utf-8

import nltk
import re
import pickle
import make_indexed_corpus as mic


def gettidy(sentences):
    sentences2 = []
    for sentence in sentences:
        sent = [word.lower() for word in sentence if re.match("\\w", word)]
        sentences2.append(sent)
    return sentences2

if __name__ == "__main__":
    min_subvocab_length = 3
    max_subvocab_length = 6
    sentences = gettidy(nltk.corpus.brown.sents())
    subvocab,vocab = mic.dictionary(sentences,min_subvocab_length, max_subvocab_length)
    code = mic.corpus_to_vocab_index(sentences,vocab)

    with open("Data/corpus","wb") as pk:
        pickle.dump(code, pk)

    with open("Data/vocab","wb") as pk:
        pickle.dump(vocab, pk)

    with open("Data/subvocab","wb") as pk:
        pickle.dump((subvocab_length, subvocab), pk)
