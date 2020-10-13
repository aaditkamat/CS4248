# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import pdb
import pickle
import os
import math
import sys
import datetime

START_TOKEN = '<s>'
END_TOKEN = '</s>'

# Set probability of OOV (Out of Vocabulary) words as 0
def handle_out_of_vocabulary(lines, model):
    words = set([word for line in lines for word in line.split(' ')])
    pos_tags = model['pos_tags']
    observation_likelihoods = model['observation_likelihoods']
    for word in words:
        if word not in observation_likelihoods:
            observation_likelihoods[word] = {}
            for pos_tag in pos_tags:
                observation_likelihoods[word][pos_tag] = 0

def calculate_alpha_values(bigram_counts, unigram_probabilities, bigram_probabilities):
    alpha_values = {}
    for prev_token in bigram_counts:
        numerator = 1 - sum([bigram_probabilities[prev_token][next_token] for next_token in bigram_counts[prev_token] if bigram_counts[prev_token][next_token] > 0])
        denominator = sum([unigram_probabilities[next_token] for next_token in bigram_counts[prev_token] if bigram_counts[prev_token][next_token] == 0])
        alpha_values[prev_token] = numerator/denominator if denominator > 0 else numerator
    return alpha_values

def discount(probabilities, additive=0.01):
    # Unigram probabilities
    if type(list(probabilities.values())[0]) == float:
        unigram_probabilities = probabilities
        count_zeroes = len([probability for probability in unigram_probabilities.values() if probability == 0 ])
        discount_factor = additive * count_zeroes / (len(unigram_probabilities.values()) - count_zeroes)
        return {unigram: probability + additive if probability == 0 else probability  - discount_factor for unigram, probability in unigram_probabilities.items()}
    # Bigram probabilities
    bigram_probabilities = probabilities
    count_zeroes = len([bigram_probabilities[prev_token][next_token] for prev_token in bigram_probabilities for next_token in bigram_probabilities[prev_token] if bigram_probabilities[prev_token][next_token] == 0 ])
    discount_factor = additive * count_zeroes / (sum([len(bigram_probabilities[prev_token].values()) for prev_token in bigram_probabilities]) - count_zeroes)
    discounted_bigram_probabilities = {}
    for prev_token in bigram_probabilities:
        discounted_bigram_probabilities[prev_token] = {}
        for next_token in bigram_probabilities[prev_token]:
            probability = bigram_probabilities[prev_token][next_token]
            new_probability = probability + additive if probability == 0 else abs(probability - discount_factor)
            discounted_bigram_probabilities[prev_token][next_token] = new_probability
    return discounted_bigram_probabilities

def smoothen(unigram_counts, bigram_counts, bigram_probabilities):
    total_number_of_unigrams = sum(unigram_counts.values())
    unigram_probabilities = {unigram: unigram_counts[unigram] / total_number_of_unigrams for unigram in unigram_counts}
    discounted_unigram_probabilities = discount(unigram_probabilities)
    discounted_bigram_probabilities = discount(bigram_probabilities)
    pdb.set_trace()
    alpha_values = calculate_alpha_values(bigram_counts, discounted_unigram_probabilities, discounted_bigram_probabilities)
    for prev_token in bigram_counts:
        for next_token in bigram_counts[prev_token]:
            if bigram_counts[prev_token][next_token] > 0:
                bigram_probabilities[prev_token][next_token] = discounted_bigram_probabilities[prev_token][next_token]
            else:
                bigram_probabilities[prev_token][next_token] = alpha_values[prev_token] * discounted_unigram_probabilities[next_token]

def preprocess(lines, model):
    handle_out_of_vocabulary(lines, model)
    word_unigram_counts, word_pos_bigram_counts, observation_likelihoods = model['word_unigram_counts'], model['word_pos_bigram_counts'], model['observation_likelihoods']
    # smoothen(word_unigram_counts, word_pos_bigram_counts, observation_likelihoods)
    pos_unigram_counts, pos_pos_bigram_counts, transition_probabilities = model['word_unigram_counts'], model['pos_pos_bigram_counts'], model['transition_probabilities']
    # smoothen(pos_unigram_counts, pos_pos_bigram_counts, transition_probabilities)

def viterbi(model, line):
    words = line.split(' ')
    transition_probabilities = model['transition_probabilities']
    observation_likelihoods = model['observation_likelihoods']
    pos_tags = model['pos_tags']
    forward_ptr = {}

    for pos_tag in pos_tags:
        forward_ptr[pos_tag] = {}
        forward_ptr[pos_tag][words[0]] = transition_probabilities[START_TOKEN][pos_tag] * observation_likelihoods[words[0]][pos_tag]

    for i in range(1, len(words)):
        for curr_pos_tag in pos_tags:          
            forward_ptr[curr_pos_tag][words[i]] = max([forward_ptr[prev_pos_tag][words[i - 1]] * transition_probabilities[prev_pos_tag][curr_pos_tag] * observation_likelihoods[words[i]][curr_pos_tag] for prev_pos_tag in pos_tags])

    forward_ptr[END_TOKEN] = {}
    forward_ptr[END_TOKEN][words[-1]] = max([forward_ptr[pos_tag][words[-1]] * transition_probabilities[pos_tag][END_TOKEN] for pos_tag in pos_tags])

    return forward_ptr

def read_test_file(test_file):
    with open(test_file) as test_file_handler:
        lines = test_file_handler.read().split('\n')
        return lines

def read_model_file(model_file):
    with open(model_file, mode='rb') as model_file_handler:
        return pickle.load(model_file_handler)

def get_pos_tags(words, pos_tags, forward_ptr):
    generated_pos_tags = []
    for word in words:
        best_tag, probability = pos_tags[0], forward_ptr[pos_tags[0]][word]
        for tag in pos_tags[1: ]:
            if forward_ptr[tag][word] > probability:
                best_tag, probability = tag, forward_ptr[tag][word]
        generated_pos_tags.append(best_tag)
    return generated_pos_tags

def write_to_output_file(lines, out_file, model):
    with open(out_file, 'a') as output_file_handler:
        ctr = 0
        for line in lines[: -1]:
            forward_ptr = viterbi(model, line)
            words = line.split(' ')

            pos_tags = model['pos_tags']
 
            word_tags = get_pos_tags(words, pos_tags, forward_ptr)
            new_line = ' '.join(['{}/{}'.format(words[i], word_tags[i]) for i in range(len(words))])
            output_file_handler.write(new_line + '\n')
            ctr += 1

def tag_sentence(test_file, model_file, out_file, start_time):
    lines = read_test_file(test_file)
    model = read_model_file(model_file)
    preprocess(lines, model)
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
