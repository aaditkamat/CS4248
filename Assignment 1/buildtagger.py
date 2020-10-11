# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
import pdb
import numpy as np
import json
import os
import math
import sys
import datetime

class HiddenMarkovModel():
    def __init__(self, T, pos_pos_bigram_counts, pos_tag_counts, word_counts, word_pos_counts):
        self.A = self.calculate_transition_probabilities(pos_pos_bigram_counts, pos_tag_counts)
        self.observations = list(word_counts)[: T]
        self.B = self.observation_likelihoods(T, word_counts, word_pos_counts, pos_tag_counts)

    def __fill_probabilities(self, size):
        # initially fill with values taken from the logistic distribution
        matrix = np.random.default_rng().logistic(size=size)
        # normalize values in the first row
        for i in range(matrix.shape[0]):
            matrix[i] = np.abs(matrix[i])
            matrix[i] = matrix[i] / np.sum(matrix[i])
        return matrix

    def calculate_transition_probabilities(self, pos_pos_bigram_counts, pos_tag_counts):
        # pdb.set_trace()
        N = len(pos_tag_counts)
        tags = list(pos_tag_counts.keys())
        matrix = self.__fill_probabilities((N + 1, N))
        for i in range(1, N + 1):
            for j in range(N):
                if tags[i - 1] in pos_pos_bigram_counts and tags[j] in pos_pos_bigram_counts[tags[i - 1]]:
                    matrix[i, j] = pos_pos_bigram_counts[tags[i - 1]][tags[j]] / pos_tag_counts[tags[i - 1]]
                else:
                    matrix[i, j] = 0
        return matrix

    def observation_likelihoods(self, T, word_counts, word_pos_counts, pos_tag_counts):
        N = len(pos_tag_counts)
        tags = list(pos_tag_counts.keys())
        # initially fill with values taken from the normal distribution
        matrix = self.__fill_probabilities((N, T))
        for i in range(N):
            for j in range(T):
                if tags[i] in word_pos_counts and tags[j] in word_pos_counts[tags[i]]:
                    matrix[i, j] = word_pos_counts[tags[i]][tags[j]] / word_counts[tags[i]]
                else:
                    matrix[i, j] = 0
        return matrix
        
def calculate_bigram_counts(bigram_counts, previous_token, current_token):
    if previous_token in bigram_counts and current_token in bigram_counts[previous_token]:
        bigram_counts[previous_token][current_token] += 1
    elif previous_token in bigram_counts:
        bigram_counts[previous_token][current_token] = 1
    else:
        bigram_counts[previous_token] = {current_token: 1}

def calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, previous_pos_tag, current_pos_tag):
    calculate_bigram_counts(pos_pos_bigram_counts, previous_pos_tag, current_pos_tag)

def calculate_word_pos_counts(word_pos_counts, current_word, current_pos_tag):
    calculate_bigram_counts(word_pos_counts, current_word, current_pos_tag)

def calculate_unigram_counts(token, counts):
    if token in counts:
        counts[token] += 1
    else:
        counts[token] = 1

def custom_split(tagged_word):
    return ('/'.join(tagged_word.split('/')[0: -1]), tagged_word.split('/')[-1])

def process_lines(lines):
    pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts = {}, {}, {}, {}
    for line in lines:
        tagged_words = line.split(' ')
        current_word, current_pos_tag = custom_split(tagged_words[0])
        calculate_word_pos_counts(word_pos_counts, current_word, current_pos_tag)
        calculate_unigram_counts(current_word, word_counts)
        calculate_unigram_counts(current_pos_tag, pos_tag_counts)
        for i in range(1, len(tagged_words)):
            current_word, current_pos_tag = custom_split(tagged_words[i])
            previous_word, previous_pos_tag = custom_split(tagged_words[i - 1])
            calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, previous_pos_tag, current_pos_tag)
            calculate_word_pos_counts(word_pos_counts, current_word, current_pos_tag)
            calculate_unigram_counts(current_word, word_counts)
            calculate_unigram_counts(current_pos_tag, pos_tag_counts)
    return pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    with open(train_file, mode='r') as train_file_handler:
        train_data = train_file_handler.read()
        lines = train_data.split('\n')
        pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts = process_lines(lines)
        hmm = HiddenMarkovModel(10, pos_pos_bigram_counts, pos_tag_counts, word_counts, word_pos_counts)
        # pdb.set_trace()
        with open(model_file, mode='w') as model_file_handler:
            json.dump({
                'pos_tags': list(pos_tag_counts),
                'transition_probabilities': hmm.A.tolist(),
                'observation_likelihoods': hmm.B.tolist()}, 
                model_file_handler)


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
