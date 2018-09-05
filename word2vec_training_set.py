# coding: utf-8

def makeframes(corpus, framelength):
    """
    Make frames from a corpus with length framelength
    """
    frames = []
    for sentence in corpus:
        for i in range(len(sentence)):
            prev_start = max(i - framelength, 0)
            prev_end = min(len(sentence), i + framelength)
            frame = [sentence[i], sentence[prev_start:i] + sentence[i+1:prev_end]]
            frames.append(frame)
    return frames

def make_skipgram_samples(frames):
    """
    Make skipgram tranining set from a frame list
    """
    return [(frame[0],context) for frame in frames for context in frame[1]]

if __name__ == "__main__":
    import pickle
    with open("Data/corpus","rb") as pk:
        corpus = pickle.load(pk)
    frames = makeframes(corpus,15)
    samples = make_skipgram_samples(frames)
    with open("Data/frames","wb") as pk:
        pickle.dump(frames, pk)
    with open("Data/skipgram_samples","wb") as pk:
        pickle.dump(samples, pk)
