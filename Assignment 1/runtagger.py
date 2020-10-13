# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import pdb
import pickle
import os
import math
import sys
import datetime

from collections import defaultdict

START_TOKEN = '<s>'
END_TOKEN = '</s>'

# Set prob of OOV (Out of Vocabulary) words as 0
# def handle_out_of_vocabulary(lines, model):
#     words = set([word for line in lines for word in line.split(' ')])
#     pos_tags = model['pos_tags']
#     emission_probs = model['emission_probs']
#     for word in words:
#         if word not in emission_probs:
#             emission_probs[word] = {}
#             for pos_tag in pos_tags:
#                 emission_probs[word][pos_tag] = 0

def calculate_alpha_values(bigram_counts, unigram_probs, bigram_probs):
    alpha_values = {}
    for prev_token in bigram_counts:
        numerator = 1 - sum([bigram_probs[prev_token][next_token] for next_token in bigram_counts[prev_token] if bigram_counts[prev_token][next_token] > 0])
        denominator = sum([unigram_probs[next_token] for next_token in bigram_counts[prev_token] if bigram_counts[prev_token][next_token] == 0])
        alpha_values[prev_token] = numerator/denominator if denominator > 0 else numerator
    return alpha_values

def discount(probs, additive=0.01):
    # Unigram probs
    if type(list(probs.values())[0]) == float:
        unigram_probs = probs
        count_zeroes = len([prob for prob in unigram_probs.values() if prob == 0 ])
        discount_factor = additive * count_zeroes / (len(unigram_probs.values()) - count_zeroes)
        return {unigram: prob + additive if prob == 0 else prob  - discount_factor for unigram, prob in unigram_probs.items()}
    # Bigram probs
    bigram_probs = probs
    count_zeroes = len([bigram_probs[prev_token][next_token] for prev_token in bigram_probs for next_token in bigram_probs[prev_token] if bigram_probs[prev_token][next_token] == 0 ])
    discount_factor = additive * count_zeroes / (sum([len(bigram_probs[prev_token].values()) for prev_token in bigram_probs]) - count_zeroes)
    discounted_bigram_probs = {}
    for prev_token in bigram_probs:
        discounted_bigram_probs[prev_token] = {}
        for next_token in bigram_probs[prev_token]:
            prob = bigram_probs[prev_token][next_token]
            new_prob = prob + additive if prob == 0 else abs(prob - discount_factor)
            discounted_bigram_probs[prev_token][next_token] = new_prob
    return discounted_bigram_probs

def smoothen(unigram_counts, bigram_counts, bigram_probs):
    total_number_of_unigrams = sum(unigram_counts.values())
    unigram_probs = {unigram: unigram_counts[unigram] / total_number_of_unigrams for unigram in unigram_counts}
    discounted_unigram_probs = discount(unigram_probs)
    discounted_bigram_probs = discount(bigram_probs)
    alpha_values = calculate_alpha_values(bigram_counts, discounted_unigram_probs, discounted_bigram_probs)
    for prev_token in bigram_counts:
        for next_token in bigram_counts[prev_token]:
            if bigram_counts[prev_token][next_token] > 0:
                bigram_probs[prev_token][next_token] = discounted_bigram_probs[prev_token][next_token]
            else:
                bigram_probs[prev_token][next_token] = alpha_values[prev_token] * discounted_unigram_probs[next_token]

# def preprocess(lines, model):
#     # handle_out_of_vocabulary(lines, model)
#     word_unigram_counts, word_pos_bigram_counts, emission_probs = model['word_unigram_counts'], model['word_pos_bigram_counts'], model['emission_probs']
#     # smoothen(word_unigram_counts, word_pos_bigram_counts, emission_probs)
#     pos_unigram_counts, pos_pos_bigram_counts, transition_probs = model['word_unigram_counts'], model['pos_pos_bigram_counts'], model['transition_probs']
#     # smoothen(pos_unigram_counts, pos_pos_bigram_counts, transition_probs)

def viterbi(model, line):
    words = line.split(' ')
    transition_probs = model['transition_probs']
    emission_probs = model['emission_probs']
    tags = model['tag_counts'].keys()
    forward_ptr = defaultdict(float)

    for tag in tags:
        forward_ptr[(tag, words[0])] = transition_probs[(START_TOKEN, tag)] * emission_probs[(tag, words[0])]

    for i in range(1, len(words)):
        word1, word2 = words[i - 1: i + 1]
        for tag2 in tags:          
            forward_ptr[(tag2, word2)] = max([forward_ptr[(tag1, word1)] * transition_probs[(tag1, tag2)] * emission_probs[(tag2, word2)] for tag1 in tags])

    forward_ptr[(END_TOKEN, words[-1])] = max([forward_ptr[(tag, words[-1])] * transition_probs[(tag, END_TOKEN)] for tag in tags])

    # pdb.set_trace()
    return forward_ptr

def read_test_file(test_file):
    with open(test_file) as test_file_handler:
        lines = test_file_handler.read().split('\n')
        return lines

# Objects obtained from the pickled file need to be converted to the custom defaultdict type 
# because they are stored as dictionaries
def restore_types(model):
    typed_model = {}
    for name, dictionary in model.items():
        new_dictionary = defaultdict(type(list(dictionary.values())[0]))
        for key in dictionary:
            new_dictionary[key] = dictionary[key]
        typed_model[name] = new_dictionary
    return typed_model

def read_model_file(model_file):
    with open(model_file, mode='rb') as model_file_handler:
        model = pickle.load(model_file_handler)
        return restore_types(model)


def get_pos_tags(words, tags, forward_ptr):
    generated_pos_tags = []
    for word in words:
        best_tag, prob = None, 0
        for tag in tags:
            if forward_ptr[(tag, word)] >= prob:
                best_tag, prob = tag, forward_ptr[(tag, word)]
        generated_pos_tags.append(best_tag)
    return generated_pos_tags

def write_to_output_file(lines, out_file, model):
    with open(out_file, 'a') as output_file_handler:
        ctr = 0
        for line in lines[: -1]:
            forward_ptr = viterbi(model, line)
            words = line.split(' ')

            tags = model['tag_counts'].keys()
 
            word_tags = get_pos_tags(words, tags, forward_ptr)
            new_line = ' '.join(['{}/{}'.format(words[i], word_tags[i]) for i in range(len(words))])
            output_file_handler.write(new_line + '\n')
            ctr += 1

def tag_sentence(test_file, model_file, out_file, start_time):
    lines = read_test_file(test_file)
    model = read_model_file(model_file)
    # preprocess(lines, model)
    write_to_output_file(lines, out_file, model)
    
if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file, start_time)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
