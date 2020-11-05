import sys
from collections import defaultdict, Counter
import math
import random
import numpy as np
import os

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
   
    Input: list[string], int
    output: list[tuple(string)]
    
    """
    if n < 1:
        return None
    
    res, s = [], sequence + ['STOP']
    
    for i in range(len(s)):
        ngram = ['START' for _ in range(n)]
        j = 0
        while j < n and i- j >= 0 : 
            ngram[n-1-j] = s[i-j]
            j += 1
        res += [tuple(ngram)] 
        
    if n == 1:
        res = [('START',)] + res
        
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.trigramdict = None


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        
        self.unigramcounts = defaultdict(int) 
        self.bigramcounts = defaultdict(int)  
        self.trigramcounts = defaultdict(int)
        
        self.totaltricounts = self.totalbicounts = self.totalunicounts = 0
        
        for sequence in corpus:
            unigram_counts = Counter(get_ngrams(sequence, 1))
            for k in unigram_counts:
                self.unigramcounts[k] += unigram_counts[k]
                self.totalunicounts += unigram_counts[k]
                
            bigram_counts = Counter(get_ngrams(sequence, 2))
            for k in bigram_counts:
                self.bigramcounts[k] += bigram_counts[k]
                self.totalbicounts += bigram_counts[k]
                
            trigram_counts = Counter(get_ngrams(sequence, 3))
            for k in trigram_counts:
                self.trigramcounts[k] += trigram_counts[k]
                self.totaltricounts += trigram_counts[k]
        
            self.bigramcounts[('START','START')] += 1
            
    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        return self.trigramcounts[trigram]*1.0/self.totaltricounts

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram]*1.0/self.totalbicounts
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        return self.unigramcounts[unigram]*1.0/self.totalunicounts

    def create_trigram_dict(self):
        self.trigramdict = defaultdict(list)
        for trigram in self.trigramcounts:
            self.trigramdict[trigram[:2]] += [trigram[2]]
      
    
    def generate_sentence(self, t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        if not self.trigramdict: 
            self.create_trigram_dict()
            
        fringe = [(('START', 'START'), [])]
        
        while fringe:
            curr, s = fringe.pop()
            if s and s[-1] == "STOP":
                return s
                
            if len(s) < t:
                nxts = [nxt for nxt in self.trigramdict[curr]]
                probs = np.array([self.raw_trigram_probability(tuple(list(curr) + [nxt])) for nxt in self.trigramdict[curr]])
                probs = probs / np.sum(probs) 
                inds = np.argsort(np.random.multinomial(200, probs))
                for i in inds:
                    if nxts[i] != "UNK":
                        fringe += [( (curr[1], nxts[i]), s+[nxts[i]])]
            
        return ""

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        
        u -> v -> w
        p_{mle}(w|u,v) = count(u,v,w) / count(u,v)
        p_{mle}(w|v) = count(v,w) / count(v)
        p_{mle}(w) = count(w) / total(unigram)  
        
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        mle_w_given_uv = self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]] if self.bigramcounts[trigram[:2]] else 0 
        mle_w_given_v = self.bigramcounts[trigram[1:]] / self.unigramcounts[trigram[1:2]] if self.unigramcounts[trigram[1:2]] else 0
        mle_w = self.unigramcounts[trigram[2:]] / self.totalunicounts
        
        return lambda1 * mle_w_given_uv + lambda2 * mle_w_given_v + lambda3 * mle_w 
              
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
                
        log_prob = 0
        trigrams = get_ngrams(sentence, 3)
        
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            log_prob += math.log2(prob) if prob else 0
           
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l, M = 0, 0
        for sequence in corpus:
            l += self.sentence_logprob(sequence)
            M += len(sequence)
        l = l*1.0 / M
            
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1) # high
        model2 = TrigramModel(training_file2) # low

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            correct += pp_1 < pp_2
            total += 1
    
        for f in os.listdir(testdir2):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            correct += pp_2 < pp_1
            total += 1
        
        
        return correct*1.0 / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    
    # Essay scoring experiment: 

    acc = essay_scoring_experiment(*sys.argv[1:])
    print('Accuracy: {}'.format(acc))

