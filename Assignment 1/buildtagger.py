# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
import pdb
import numpy as np
import torch
import os
import math
import sys
import datetime

class HiddenMarkovModel(torch.nn.Module):
    def __init__(self, D_in, T, H, D_out, pos_pos_bigram_counts, word_pos_counts):
        super(HiddenMarkovModel, self).__init__()
        self.input_layer = torch.nn.Linear(D_in, H)
        self.hidden_layers = [torch.nn.Linear(H, H) for _ in range(T - 1)]
        self.output_layer = torch.nn.Linear(H, D_out)
        self.A = self.calculate_transition_probabilities(pos_pos_bigram_counts)
        self.B = self.observation_likelihoods(word_pos_counts)
        self.viterbi_tensor = self.viterbi(T, H)

    def calculate_transition_probabilities(self, pos_pos_bigram_counts):
        pass

    def observation_likelihoods(self, word_pos_counts):
        pass

    def viterbi(self, T, H):
        viterbi_tensor = torch.zeros([T + 2, H], dtype=torch.int32)
        pass

    def forward(self, x):
        return viterbi_tensor

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

def process_lines(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts, lines):
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

def create_model(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts):
    # hyperparameters for our model: D_in is the input dimension, H is the hidden dimension
    # D_out is the output_dimension and N is the number of hidden lyaers corresponding to the
    # number of words to be pos tagged
    D_in, N, H, D_out = 1, 10, 45, 1
    learning_rate = 1e-4
    iterations = 100

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = HiddenMarkovModel(D_in, N, H, D_out, pos_pos_bigram_counts, word_pos_counts)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for iteration in range(iterations):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print('Iteration {}: Loss: {}'.format(iteration, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    with open(train_file) as file:
        pos_pos_bigram_counts = {}
        word_pos_counts = {}
        word_counts = {}
        pos_tag_counts = {}
        train_data = file.read()
        lines = train_data.split('\n')
        process_lines(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts, lines)
        create_model(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts)
        pdb.set_trace()


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
