import numpy as np
import pandas as pd
from collections import Counter



class NGramModel:
    def __init__(self, n, laplace_smoothing=0, min_token_counts=3):
        self.n = n
        self.vocab_counts = {}
        self.vocab = set()
        self.n_grams = {}
        self.n_minus_grams = {}
        self.laplace_smoothing = laplace_smoothing
        self.min_token_counts=min_token_counts
        
    
    def fit(self, inputs):
        # inputs.shape = [str, str, ...]
        df = pd.DataFrame(inputs ,columns=['sentence'])
        
        df['sentence'] = "<START> "*(self.n-1)+df['sentence'].str.replace("\n", "") + " <STOP>"
        
        whole_string = " ".join(df['sentence'].tolist()).split(" ")
        vocab_counts  = dict(Counter(whole_string))
        
        unk_token_counts = sum(counts for counts in vocab_counts.values() if counts < self.min_token_counts)
        vocab_counts = {k:v for k,v in vocab_counts.items() if v>=self.min_token_counts}
        vocab_counts.update({"<UNK>": unk_token_counts})
        self.vocab_counts=vocab_counts
        
        # self.vocab_to_index = {k:i for i, (k,v) in enumerate(self.vocab_counts.items())}
        # self.index_to_vocab = {v:k for k,v in self.vocab_to_index.items()}

        df['sentence'] = df['sentence'].str.split(' ')
        df['sentence'] = df['sentence'].apply(lambda sentence: [token if token in self.vocab_counts else "<UNK>" for token in sentence])
        df['sentence'].apply(lambda x: self.generate_ngram_posteriors(x, self.n))
        # self.vocab = [word for word in self.vocab_counts.keys() if word!="<START>"]
        
        
    def generate_ngram_posteriors(self, values_list, n):
        for i in range(len(values_list)-n+1):
            n_gram = tuple(values_list[i:i+n])
            if n_gram not in self.n_grams:
                self.n_grams[n_gram]=1
            else:
                self.n_grams[n_gram]+=1
            
            if self.n>1:
            
                if n_gram[:-1] not in self.n_minus_grams:
                    self.n_minus_grams[n_gram[:-1]] = 1
                else:
                    self.n_minus_grams[n_gram[:-1]]+=1
            
            self.vocab.add(n_gram[-1])
                
    
    @staticmethod
    def generate_ngrams(values_list, n):
        ngrams = [tuple(values_list[i:i+n]) for i in range(len(values_list)-n+1)]
        return ngrams
    
    
    def compute_prob_for_ngram(self, n_gram):
        n_gram_counts = self.n_grams.get(n_gram, 0)
        n_minus_gram_counts = self.n_minus_grams.get(n_gram[:-1], 0) if self.n>1 else len(self.vocab)
        if n_gram_counts==0:
            pass
        if n_minus_gram_counts==0:
            pass
        if self.laplace_smoothing>0:
            return (n_gram_counts+self.laplace_smoothing)/(n_minus_gram_counts + len(self.vocab))
        else:
            return n_gram_counts/n_minus_gram_counts
    
    
    def predict_probs_for_sentence(self, sentence):
        sentence = "<START> "*(self.n-1)+sentence.replace("\n", "") + " <STOP>"
        sentence = sentence.split(' ')
        sentence = [token if token in self.vocab_counts else "<UNK>" for token in sentence]
        ngrams = self.generate_ngrams(sentence, self.n)
        return [self.compute_prob_for_ngram(ngram) for ngram in ngrams]
    
    
    def compute_perplexity(self, inputs):
        log_probs = []
        for sentence in inputs:
            log_probs.extend(self.predict_probs_for_sentence(sentence))
        
        log_probs = np.log2(log_probs)
        perplexity = np.exp2(-np.mean(log_probs))
        return perplexity
    

    # function to test if values in n_grams sum to 1 for a given n-1 gram
    def test_n_grams(self):
        for key in self.n_minus_grams.keys():
            n_grams = [self.n_grams[k] for k in self.n_grams.keys() if k[:-1]==key]
            total = sum(n_grams)
            assert total==self.n_minus_grams[key]
        return True


if __name__=="__main__":
    
    with open('A2-Data/1b_benchmark.train.tokens', 'r') as f:
        train_data = f.readlines()
    
    with open('A2-Data/1b_benchmark.dev.tokens', 'r') as f:
        dev_data = f.readlines()
    
    
    model = NGramModel(3, laplace_smoothing=1, min_token_counts=3)
    model.fit(train_data)
    print(model.predict_probs_for_sentence("HDTV ."))
    print(model.compute_perplexity(dev_data))
    