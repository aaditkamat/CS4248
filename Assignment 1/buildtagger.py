# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
import pdb
import pickle
import os
import math
import sys
import datetime

START_TOKEN = '<s>'
END_TOKEN = '</s>'

class HiddenMarkovModel():
    def __init__(self, pos_pos_bigram_counts, pos_unigram_counts, word_unigram_counts, word_pos_bigram_counts):
        self.transition_probabilities = self.calculate_transition_probabilities(pos_pos_bigram_counts, pos_unigram_counts)
        self.observation_likelihoods = self.calculate_observation_likelihoods(word_unigram_counts, word_pos_bigram_counts, pos_unigram_counts)

    def calculate_transition_probabilities(self, pos_pos_bigram_counts, pos_unigram_counts):
        pos_tags = set(pos_unigram_counts.keys())
        probabilities = {}
        for first_tag in pos_tags:
            probabilities[first_tag] = {}
            for second_tag in pos_tags:
                if first_tag in pos_pos_bigram_counts and second_tag in pos_pos_bigram_counts[first_tag]:
                    probabilities[first_tag][second_tag] = pos_pos_bigram_counts[first_tag][second_tag] / pos_unigram_counts[first_tag]
                else:
                    probabilities[first_tag][second_tag] = 0
        return probabilities

    def calculate_observation_likelihoods(self, word_unigram_counts, word_pos_bigram_counts, pos_unigram_counts):
        pos_tags = set(pos_unigram_counts.keys())
        # actual words in the training file
        words = set(word_unigram_counts.keys()) - set([START_TOKEN, END_TOKEN])
        probabilities = {}
        for word in words:
            probabilities[word] = {}
            for pos_tag in pos_tags:
                if word in word_pos_bigram_counts and pos_tag in word_pos_bigram_counts[word]:
                    probabilities[word][pos_tag] = word_pos_bigram_counts[word][pos_tag] / word_unigram_counts[word]
                else:
                    probabilities[word][pos_tag] = 0
        return probabilities
        
def calculate_bigram_counts(bigram_counts, previous_token, current_token):
    if previous_token in bigram_counts and current_token in bigram_counts[previous_token]:
        bigram_counts[previous_token][current_token] += 1
    elif previous_token in bigram_counts:
        bigram_counts[previous_token][current_token] = 1
    else:
        bigram_counts[previous_token] = {current_token: 1}

def calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, previous_pos, current_pos):
    calculate_bigram_counts(pos_pos_bigram_counts, previous_pos, current_pos)

def calculate_word_pos_bigram_counts(word_pos_bigram_counts, current_word, current_pos):
    calculate_bigram_counts(word_pos_bigram_counts, current_word, current_pos)

def calculate_unigram_counts(token, counts):
    if token in counts:
        counts[token] += 1
    else:
        counts[token] = 1

def custom_split(tagged_word):
    return ('/'.join(tagged_word.split('/')[0: -1]), tagged_word.split('/')[-1])

def process_train_file(train_file):
    with open(train_file, 'r') as train_file_handler:
        pos_pos_bigram_counts, word_pos_bigram_counts, word_unigram_counts, pos_unigram_counts = {}, {}, {}, {}
        train_data = train_file_handler.read()
        lines = train_data.split('\n')
        for line in lines:
            tagged_words = line.split(' ')
            calculate_unigram_counts(START_TOKEN, word_unigram_counts)
            calculate_unigram_counts(START_TOKEN, pos_unigram_counts)
            for i in range(len(tagged_words)):
                current_word, current_pos = custom_split(tagged_words[i])
                calculate_word_pos_bigram_counts(word_pos_bigram_counts, current_word, current_pos)
                calculate_unigram_counts(current_word, word_unigram_counts)
                calculate_unigram_counts(current_pos, pos_unigram_counts)
                # use start token at the beginning of a sentence as a pos tag
                if i == 0:
                    calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, START_TOKEN, current_pos)
                    calculate_word_pos_bigram_counts(word_pos_bigram_counts, current_word, START_TOKEN)
                else:
                    previous_word, previous_pos = custom_split(tagged_words[i - 1])
                    calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, previous_pos, current_pos)
            calculate_unigram_counts(END_TOKEN, word_unigram_counts)
            calculate_unigram_counts(END_TOKEN, pos_unigram_counts)
            calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, current_pos, END_TOKEN)
        return pos_pos_bigram_counts, word_pos_bigram_counts, word_unigram_counts, pos_unigram_counts

def write_to_model_file(model_file, model):
    with open(model_file, mode='wb') as model_file_handler:
            pickle.dump(model, model_file_handler)

def train_model(train_file, model_file):
    pos_pos_bigram_counts, word_pos_bigram_counts, word_unigram_counts, pos_unigram_counts = process_train_file(train_file)
    hmm = HiddenMarkovModel(pos_pos_bigram_counts, pos_unigram_counts, word_unigram_counts, word_pos_bigram_counts)
    model = {
        'pos_tags': list(set(pos_unigram_counts.keys()) - set([START_TOKEN, END_TOKEN])),
        'word_unigram_counts': word_unigram_counts,
        'pos_unigram_counts': pos_unigram_counts,
        'pos_pos_bigram_counts': pos_pos_bigram_counts,
        'word_pos_bigram_counts': word_pos_bigram_counts,
        'transition_probabilities': hmm.transition_probabilities,
        'observation_likelihoods': hmm.observation_likelihoods
    }
    # pdb.set_trace()
    write_to_model_file(model_file, model)


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
