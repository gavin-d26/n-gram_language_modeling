import numpy as np
import pandas as pd
from collections import Counter
import time

output_path = "tasks_output_final.txt"


class NGramModel:
    def __init__(self, n, laplace_smoothing=0, min_token_counts=3, eps=1e-9):
        self.n = n
        self.vocab_counts = {}
        self.vocab = set()
        self.n_grams = {}
        self.n_minus_grams = {}
        self.laplace_smoothing = laplace_smoothing
        self.min_token_counts = min_token_counts
        self.corpus_size = 0
        self.eps = eps

    # function to fit the model on the input data
    def fit(self, inputs):
        """computes n-gram posteriors for the input data

        Args:
            inputs (list): list of strings
        """

        df = pd.DataFrame(inputs, columns=["sentence"])
        df["sentence"] = (
            "<START> " * (self.n - 1) + df["sentence"].str.replace("\n", "") + " <STOP>"
        )

        whole_string = " ".join(df["sentence"].tolist()).split(" ")
        vocab_counts = dict(Counter(whole_string))

        unk_token_counts = sum(
            counts for counts in vocab_counts.values() if counts < self.min_token_counts
        )
        vocab_counts = {
            k: v for k, v in vocab_counts.items() if v >= self.min_token_counts
        }
        vocab_counts.update({"<UNK>": unk_token_counts})
        self.vocab_counts = vocab_counts

        df["sentence"] = df["sentence"].str.split(" ")
        df["sentence"] = df["sentence"].apply(
            lambda sentence: [
                token if token in self.vocab_counts else "<UNK>" for token in sentence
            ]
        )
        df["sentence"].apply(lambda x: self.generate_ngram_posteriors(x, self.n))

        for k, v in self.vocab_counts.items():
            if k != "<START>":
                self.corpus_size += v

    def generate_ngram_posteriors(self, values_list, n):
        """generates n-gram posteriors and n-1 gram posteriors for a given list of values

        Args:
            values_list (list): list of tokens [token1, token2, ...]
            n (int): n-gram value
        """
        for i in range(len(values_list) - n + 1):
            n_gram = tuple(values_list[i : i + n])
            if n_gram not in self.n_grams:
                self.n_grams[n_gram] = 1
            else:
                self.n_grams[n_gram] += 1

            if self.n > 1:

                if n_gram[:-1] not in self.n_minus_grams:
                    self.n_minus_grams[n_gram[:-1]] = 1
                else:
                    self.n_minus_grams[n_gram[:-1]] += 1

            self.vocab.add(n_gram[-1])

    @staticmethod
    def generate_ngrams(values_list, n):
        """generates n-grams for a given list of tokens

        Args:
            values_list (list): list of tokens [token1, token2, ...]
            n (int): n-gram value

        Returns:
            List: list of n-grams like [(token1, token2, ...), (token2, token3, ...), ...]
        """
        ngrams = [
            tuple(values_list[i : i + n]) for i in range(len(values_list) - n + 1)
        ]
        return ngrams

    # function to compute probability for a given n-gram
    def compute_prob_for_ngram(self, n_gram):
        n_gram_counts = self.n_grams.get(n_gram, 0)
        n_minus_gram_counts = (
            self.n_minus_grams.get(n_gram[:-1], 0) if self.n > 1 else self.corpus_size
        )
        if n_gram_counts == 0:
            pass
        if n_minus_gram_counts == 0:
            pass
        if self.laplace_smoothing > 0:
            return (n_gram_counts + self.laplace_smoothing + self.eps) / (
                n_minus_gram_counts + len(self.vocab) * self.laplace_smoothing
            ) + self.eps
        else:
            try:
                return (n_gram_counts + self.eps) / (n_minus_gram_counts)
            except ZeroDivisionError:
                return self.eps

    def predict_probs_for_sentence(self, sentence):
        """computes probabilities for each word in the sentence using n-gram model. eg. P(w1|w2, w3) for trigram model.
        does not include <START> token in the sentence

        Args:
            sentence (str): input sentence

        Returns:
            _type_: list of probabilities for each word in the sentence
        """
        sentence = "<START> " * (self.n - 1) + sentence.replace("\n", "") + " <STOP>"
        sentence = sentence.split(" ")
        sentence = [
            token if token in self.vocab_counts else "<UNK>" for token in sentence
        ]
        ngrams = self.generate_ngrams(sentence, self.n)
        return [self.compute_prob_for_ngram(ngram) for ngram in ngrams]

    # applies predict_probs_for_sentence for each sentence in the input list
    def compute_probs_for_corpus(self, inputs):
        probs = []
        for sentence in inputs:
            probs.extend(self.predict_probs_for_sentence(sentence))
        return probs

    # computes perplexity for the input list of probabilities
    @staticmethod
    def compute_perplexity(probs):
        log_probs = np.log2(probs)
        perplexity = np.exp2(-np.mean(log_probs))
        return perplexity

    # function to test if values in n_grams sum to 1 for a given n-1 gram
    def test_n_grams(self):
        for key in self.n_minus_grams.keys():
            n_grams = [self.n_grams[k] for k in self.n_grams.keys() if k[:-1] == key]
            total = sum(n_grams)
            assert total == self.n_minus_grams[key]
        return True


class InterpolatedNGramModel:
    """class to interpolate n-gram models"""

    # initialize with models and weights, weights should sum to 1, models -> [model1, model2, ...], weights -> [w1, w2, ...]
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def compute_probs_for_corpus(self, sentences):
        probs = []
        for model in self.models:
            probs.append(model.compute_probs_for_corpus(sentences))
        probs = np.array(probs)
        probs = np.average(probs, axis=0, weights=self.weights)
        return probs

    def compute_perplexity(self, probs):
        log_probs = np.log2(probs)
        perplexity = np.exp2(-np.mean(log_probs))
        return perplexity


if __name__ == "__main__":

    output_path = open(output_path, "w")

    start = time.time()

    with open("A2-Data/1b_benchmark.train.tokens", "r") as f:
        train_data = f.readlines()

    with open("A2-Data/1b_benchmark.dev.tokens", "r") as f:
        dev_data = f.readlines()

    with open("A2-Data/1b_benchmark.test.tokens", "r") as f:
        test_data = f.readlines()

    # train unigram model
    unigram = NGramModel(1, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    unigram.fit(train_data)
    print(
        "unigram - perplexity: ",
        unigram.compute_perplexity(unigram.compute_probs_for_corpus(["HDTV ."])),
        file=output_path,
    )

    # train bigram model
    bigram = NGramModel(2, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    bigram.fit(train_data)
    print(
        "bigram - perplexity: ",
        bigram.compute_perplexity(bigram.compute_probs_for_corpus(["HDTV ."])),
        file=output_path,
    )

    # train trigram model
    trigram = NGramModel(3, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    trigram.fit(train_data)
    print(
        "trigram - perplexity: ",
        trigram.compute_perplexity(trigram.compute_probs_for_corpus(["HDTV ."])),
        file=output_path,
    )

    # test unigram model on train data and dev data
    print(
        "unigram - train perplexity: ",
        unigram.compute_perplexity(unigram.compute_probs_for_corpus(train_data)),
        file=output_path,
    )
    print(
        "unigram - dev perplexity: ",
        unigram.compute_perplexity(unigram.compute_probs_for_corpus(dev_data)),
        file=output_path,
    )

    print(
        "unigram - test perplexity: ",
        unigram.compute_perplexity(unigram.compute_probs_for_corpus(test_data)),
        file=output_path,
    )

    # test bigram model on train data and dev data and test data
    print(
        "bigram - train perplexity: ",
        bigram.compute_perplexity(bigram.compute_probs_for_corpus(train_data)),
        file=output_path,
    )
    print(
        "bigram - dev perplexity: ",
        bigram.compute_perplexity(bigram.compute_probs_for_corpus(dev_data)),
        file=output_path,
    )

    print(
        "bigram - test perplexity: ",
        bigram.compute_perplexity(bigram.compute_probs_for_corpus(test_data)),
        file=output_path,
    )

    # test trigram model on train data and dev data and test data
    print(
        "trigram - train perplexity: ",
        trigram.compute_perplexity(trigram.compute_probs_for_corpus(train_data)),
        file=output_path,
    )
    print(
        "trigram - dev perplexity: ",
        trigram.compute_perplexity(trigram.compute_probs_for_corpus(dev_data)),
        file=output_path,
    )

    print(
        "trigram - test perplexity: ",
        trigram.compute_perplexity(trigram.compute_probs_for_corpus(test_data)),
        file=output_path,
    )

    # define a list of lambda values
    lambdas = [
        [0.1, 0.3, 0.6],
        [0.3, 0.3, 0.4],
        [0.5, 0.3, 0.2],
        [0.1, 0.2, 0.7],
        [0.2, 0.2, 0.6],
    ]

    # for each lambda value, train an interpolated model and compute perplexity on the sentence "HDTV .", the training data and the dev data
    # also get best lambda values based on dev perplexity
    min_dev_perplexity = float("inf")
    for i, lambda_val in enumerate(lambdas):
        interpo_model = InterpolatedNGramModel([unigram, bigram, trigram], lambda_val)

        print("-" * 50, file=output_path)
        print(f"lambda values: {lambda_val}", file=output_path)
        probs = interpo_model.compute_probs_for_corpus(["HDTV ."])
        print(
            f"interpolated - {i+1} 'HDTV .' perplexity: ",
            interpo_model.compute_perplexity(probs),
            file=output_path,
        )

        probs = interpo_model.compute_probs_for_corpus(train_data)
        print(
            f"interpolated - {i+1} train perplexity: ",
            interpo_model.compute_perplexity(probs),
            file=output_path,
        )

        probs = interpo_model.compute_probs_for_corpus(dev_data)
        dev_perplexity = interpo_model.compute_perplexity(probs)
        print(
            f"interpolated - {i+1} dev perplexity: ",
            dev_perplexity,
            file=output_path,
        )

        if dev_perplexity < min_dev_perplexity:
            min_dev_perplexity = dev_perplexity
            best_lambda = lambda_val

    # print best lambda values
    print("-" * 50, file=output_path)
    print(f"best lambda values: {best_lambda}", file=output_path)

    # train interpolated model with best lambda values
    interpo_model = InterpolatedNGramModel([unigram, bigram, trigram], best_lambda)
    # compute perplexity on test data
    test_perplexity = interpo_model.compute_perplexity(
        interpo_model.compute_probs_for_corpus(test_data)
    )
    print(
        f"interpolated - test perplexity with best lambda: {test_perplexity}",
        file=output_path,
    )

    # Q) if you use half of the training data, would it increase or decrease the perplexity on previously unseen data?
    # Why? Provide empirical experimental evidence if necessary.

    print("-" * 50, file=output_path)

    # train unigram model with half of the training data
    unigram = NGramModel(1, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    unigram.fit(train_data[: len(train_data) // 2])

    # train bigram model with half of the training data
    bigram = NGramModel(2, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    bigram.fit(train_data[: len(train_data) // 2])

    # train trigram model with half of the training data
    trigram = NGramModel(3, laplace_smoothing=0, min_token_counts=3, eps=1e-6)
    trigram.fit(train_data[: len(train_data) // 2])

    # train interpolated model with best lambda values
    interpo_model = InterpolatedNGramModel([unigram, bigram, trigram], best_lambda)

    # compute perplexity on test data
    test_perplexity = interpo_model.compute_perplexity(
        interpo_model.compute_probs_for_corpus(test_data)
    )
    print(
        f"interpolated - test perplexity with best lambda and half training data: {test_perplexity}",
        file=output_path,
    )

    # Q) if you convert all tokens that appeared less then 5 times to <unk> (a special symbol for out-of-vocabulary
    # tokens), would it increase or decrease the perplexity on the previously unseen data compared to an ap-
    # proach that converts only a fraction of words that appeared just once to <unk>? Why? Provide empirical
    # experimental evidence if necessary.

    print("-" * 50, file=output_path)

    # train unigram model with min_token_counts = 5
    unigram = NGramModel(1, laplace_smoothing=0, min_token_counts=5, eps=1e-6)
    unigram.fit(train_data)

    # train bigram model with min_token_counts = 5
    bigram = NGramModel(2, laplace_smoothing=0, min_token_counts=5, eps=1e-6)
    bigram.fit(train_data)

    # train trigram model with min_token_counts = 5
    trigram = NGramModel(3, laplace_smoothing=0, min_token_counts=5, eps=1e-6)
    trigram.fit(train_data)

    # train interpolated model with best lambda values
    interpo_model = InterpolatedNGramModel([unigram, bigram, trigram], best_lambda)
    # compute perplexity on test data
    test_perplexity = interpo_model.compute_perplexity(
        interpo_model.compute_probs_for_corpus(test_data)
    )
    print(
        f"interpolated - test perplexity with best lambda and min_token_counts = 5: {test_perplexity}",
        file=output_path,
    )

    output_path.close()
