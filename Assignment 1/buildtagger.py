import pickle
import sys
import datetime

from collections import defaultdict
from itertools import product

START_TOKEN = '<s>'
END_TOKEN = '</s>'


class HiddenMarkovModel():
    def __init__(self, word_counts, tag_counts, bigram_counts):
        self.word_counts = word_counts
        self.tag_counts = tag_counts
        self.tags = list(self.tag_counts.keys())
        self.words = list(self.word_counts.keys())
        self.bigram_counts = bigram_counts
        self.transition_probs, self.emission_probs = defaultdict(
            float), defaultdict(float)
        self.transition_probs = self.calculate_transition_probs(
            tag_counts, bigram_counts)
        self.emission_probs = self.calculate_emission_probs(
            word_counts, tag_counts, bigram_counts)

    def calculate_probs(self, bigrams):
        probs = {}
        for bigram in bigrams:
            if self.tag_counts[bigram[0]] > 0:
                probs[bigram] = self.bigram_counts[bigram] / \
                    self.tag_counts[bigram[0]]
            else:
                probs[bigram] = 0
        return probs

    def calculate_transition_probs(self, tag_counts, bigram_counts):
        bigrams = product(self.tags, self.tags)
        return self.calculate_probs(bigrams)

    def calculate_emission_probs(self, word_counts, tag_counts, bigram_counts):
        bigrams = product(self.tags, self.words)
        return self.calculate_probs(bigrams)


def get_word_tag_pairs(line):
    return [('/'.join(token.split('/')[0: -1]),
             token.split('/')[-1]) for token in line.split(' ')]


def process_train_file(train_file):
    with open(train_file, 'r') as train_file_handler:
        bigram_counts, word_counts, tag_counts = defaultdict(
            int), defaultdict(int),  defaultdict(int)
        train_data = train_file_handler.read()
        lines = train_data.split('\n')

        for line in lines:
            word_tag_pairs = get_word_tag_pairs(line)

            for i in range(len(word_tag_pairs)):
                current_word, current_tag = word_tag_pairs[i]
                word_counts[current_word] += 1
                tag_counts[current_tag] += 1

                bigram_counts[(current_tag, current_word)] += 1
                # handle special start token seperately
                if i == 0:
                    bigram_counts[(START_TOKEN, current_tag)] += 1
                    bigram_counts[(current_word, START_TOKEN)] += 1
                else:
                    previous_word, previous_tag = word_tag_pairs[i - 1]
                    bigram_counts[(previous_tag, current_tag)] += 1

            # handle special end token seperately
            bigram_counts[(current_tag, END_TOKEN)] += 1

        # For convenience in calculating probabilities
        tag_counts[START_TOKEN] = len(lines)
        tag_counts[END_TOKEN] = len(lines)
        return word_counts, tag_counts, bigram_counts


def write_to_model_file(model_file, model):
    with open(model_file, mode='wb') as model_file_handler:
        pickle.dump(model, model_file_handler)


def train_model(train_file, model_file):
    word_counts, tag_counts, bigram_counts = process_train_file(train_file)
    hmm = HiddenMarkovModel(word_counts, tag_counts, bigram_counts)

    # Remove after calculating probabilities
    del tag_counts[START_TOKEN]
    del tag_counts[END_TOKEN]

    model = {
        'word_counts': word_counts,
        'tag_counts': tag_counts,
        'bigram_counts': bigram_counts,
        'transition_probs': hmm.transition_probs,
        'emission_probs': hmm.emission_probs
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
