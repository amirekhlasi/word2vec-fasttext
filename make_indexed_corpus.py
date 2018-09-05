
# coding: utf-8



def subwords(word, subwordlength):
    """
    return the subwords of a word with length subword.
    e.t.c "hello",3 -> ["<he", "hel", "ell", "llo","lo>"]
    """
    w = "<" + word + ">"
    if(len(w) < subwordlength):
        return([])
    sw = []
    for i in range(len(w) - subwordlength + 1):
        sw.append(w[i:i+subwordlength])
    return(sw)

def range_subwords(word, minlength, maxlength):
    """
    return the subwords of a word with length between min and max subwrodlength
    """
    return [sw for i in range(minlength, maxlength + 1) for sw in subwords(word, i)]



def dictionary(sentences, min_subwordlength, max_subwordlength):
    """
    return dictionory of all subwords and words of a corpus that is list of list of words.
    """
    words = set([word for sentence in sentences for word in sentence])
    vocab = {word: i for i, word in enumerate(words)}
    sub_words = set([sw for word in words for sw in range_subwords(word, min_subwordlength, max_subwordlength)])
    subvocab = {sw: i for i, sw in enumerate(sub_words)}
    return subvocab, vocab



def corpus_to_vocab_index(sentences, vocab):
    """
    Return a list of coded sentences that each item of sentence is index of word.
    Ignore uknown words
    """
    return [[vocab[word] for word in sentence if word in vocab] for sentence in sentences]


def corpus_to_subvocab_index(sentences,subvocab, min_subwordlength, max_subwordlength):
    """
    Return a list of coded sentences that each item of sentence is list of indexes of subwords.
    Ignore uknown subwords
    """
    return [[[subvocab[subword] for subword in range_subwords(word, min_subwordlength, max_subwordlength) if subword in subvocab]
            for word in sentence] for sentence in sentences]
