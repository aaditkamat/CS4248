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

def calc_alpha_values(bigram_counts, unigram_probs, bigram_probs):
    alpha_values = {}
    for bigram in bigram_counts:
        numerator = 1 - sum([bigram_probs[bigram] for bigram in bigram_counts if bigram_counts[bigram] > 0])
        denominator = sum([unigram_probs[bigram[0]] for bigram in bigram_counts if bigram_counts[bigram] == 0])
        alpha_values[bigram[0]] = numerator/denominator if denominator > 0 else numerator
    return alpha_values

def calc_disc_probs(unigram_counts, bigram_counts):
    total_unigram_counts, total_bigram_counts = len(unigram_counts), len(bigram_counts)
    disc_unigram_probs = {unigram: (unigram_count + 1) / total_unigram_counts for unigram, unigram_count in unigram_counts.items()}
    disc_bigram_probs = {bigram: (bigram_count + 1) / (unigram_counts[bigram[0]]) for bigram, bigram_count in bigram_counts.items()}
    return disc_unigram_probs, disc_bigram_probs

def smoothen(unigram_counts, bigram_counts, bigram_probs):
    total_number_of_unigrams = sum(unigram_counts.values())
    unigram_probs = {unigram: unigram_counts[unigram] / total_number_of_unigrams for unigram in unigram_counts}

    disc_unigram_probs, disc_bigram_probs = calc_disc_probs(unigram_counts, bigram_counts)

    bigram_probs = disc_bigram_probs
    
    # alpha_values = calc_alpha_values(bigram_counts, disc_unigram_probs, disc_bigram_probs)

    # for bigram in bigram_probs:
    #     if bigram_counts[bigram] > 0:
    #         bigram_probs[bigram] = disc_bigram_probs[bigram]
    #     else:
    #         bigram_probs[bigram] = alpha_values[bigram[0]] * disc_unigram_probs[bigram[1]]

def preprocess(lines, model):
    word_counts, bigram_counts, emission_probs = model['word_counts'], model['bigram_counts'], model['emission_probs']
    smoothen(word_counts, bigram_counts, emission_probs)
    tag_counts, transition_probs = model['word_counts'], model['transition_probs']
    smoothen(tag_counts, bigram_counts, transition_probs)

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
    with open(out_file, 'w') as output_file_handler:
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
