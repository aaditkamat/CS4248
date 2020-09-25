# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime


def calculate_pos_pos_bigram_counts(pos_pos_bigram_counts, previous_pos_tag, current_pos_tag):
    pos_pos_bigram = f'{previous_pos_tag}{current_pos_tag}'
    if pos_pos_bigram in pos_pos_bigram_counts:
        pos_pos_bigram_counts[pos_pos_bigram] += 1
    else:
        pos_pos_bigram_counts[pos_pos_bigram] = 1

def calculate_word_pos_counts(word_pos_counts, current_word, current_pos_tag):
    if current_word in word_pos_counts and current_pos_tag in word_pos_counts[current_word]:
        word_pos_counts[current_word][current_pos_tag] += 1
    elif current_word in word_pos_counts:
        word_pos_counts[current_word][current_pos_tag] = 1
    else:
        word_pos_counts[current_word] = {current_pos_tag: 1}

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

def calculate_backoff_weight(pos_pos_bigram_counts, pos_tag_counts, first_pos_tag, second_pos_tag):
    return 0.4 # TODO: Replace with actual computed value

# TODO: Vary discount in order to improve accuracy if necessary. Generally the value of discount lies between 0 and 1
def kneser_ney(pos_pos_bigram_counts, pos_tag_counts, first_pos_tag, second_pos_tag, discount=0.5):
    if f'{first_pos_tag}{second_pos_tag}' in pos_pos_bigram_counts:
        return (pos_pos_bigram_counts[f'{first_pos_tag}{second_pos_tag}'] - discount) / (pos_tag_counts[f'{first_pos_tag}'])
    else:
        alpha = calculate_backoff_weight(pos_pos_bigram_counts, pos_tag_counts, first_pos_tag, second_pos_tag)
        numerator = len([pos_tag for pos_tag in pos_tag_counts if f'{pos_tag}{second_pos_tag}' in pos_pos_bigram_counts])
        denominator = sum([len([pos_tag for pos_tag in pos_tag_counts if f'{pos_tag}{second_pos_tag}' in pos_pos_bigram_counts]) for another_pos_tag in pos_tag_counts])
        return alpha * numerator / denominator

def calculate_kneser_ney_smoothed_probabilities(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts):
    transition_probabilities = [kneser_ney(pos_pos_bigram_counts, pos_tag_counts, first_pos_tag, second_pos_tag) for first_pos_tag in pos_tag_counts for second_pos_tag in pos_tag_counts]
    return transition_probabilities

def print_debug_info(transition_probabilities):
    print('Debugging info: ')
    print(f'{transition_probabilities}')

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
        transition_probabilities = calculate_kneser_ney_smoothed_probabilities(pos_pos_bigram_counts, word_pos_counts, word_counts, pos_tag_counts)
        print_debug_info(transition_probabilities)


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
