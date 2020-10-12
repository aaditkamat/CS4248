# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import pdb
import numpy as np
import os
import json
import math
import sys
import datetime

START_TOKEN = '<s>'
END_TOKEN = '</s>'

def viterbi(transition_probabilities, observation_likelihoods, pos_tags, line):
    # Set probability of OOV (Out of Vocabulary) words as 0
    def handle_out_of_vocabulary(word):
        if word not in observation_likelihoods:
            observation_likelihoods[word] = {}
            for pos_tag in pos_tags:
                observation_likelihoods[word][pos_tag] = 0

    words = line.split(' ')
    forward_ptr = {}

    for pos_tag in pos_tags:
        forward_ptr[pos_tag], back_ptr[pos_tag] = {}, {}
        handle_out_of_vocabulary(words[0])
        forward_ptr[pos_tag][words[0]] = transition_probabilities[START_TOKEN][pos_tag] * observation_likelihoods[words[0]][pos_tag]
        back_ptr[pos_tag][words[0]] = START_TOKEN

    for i in range(1, len(words)):
        handle_out_of_vocabulary(words[i])
        for curr_pos_tag in pos_tags:          
            forward_ptr[curr_pos_tag][words[i]] = max([forward_ptr[prev_pos_tag][words[i - 1]] * transition_probabilities[prev_pos_tag][curr_pos_tag] * observation_likelihoods[words[i]][curr_pos_tag] for prev_pos_tag in pos_tags])
            back_ptr[curr_pos_tag][words[i]] = most_probable_tag(forward_ptr, curr_pos_tag, words[i - 1])

    forward_ptr[END_TOKEN] = {}
    forward_ptr[END_TOKEN][words[-1]] = max([forward_ptr[pos_tag][words[-1]] * transition_probabilities[pos_tag][END_TOKEN] for pos_tag in pos_tags])

    return forward_ptr

def process_test_file(test_file):
    with open(test_file) as test_file_handler:
        lines = test_file_handler.read().split('\n')
        return lines

def process_model_file(model_file):
    with open(model_file) as model_file_handler:
        model = json.load(model_file_handler)
        return model["pos_tags"], model["transition_probabilities"], model["observation_likelihoods"]

def get_pos_tags(words, pos_tags, forward_ptr):
    generated_pos_tags = []
    for word in words:
        best_tag, probability = pos_tags[0], forward_ptr[pos_tags[0]][word]
        for tag in pos_tags[1: ]:
            if forward_ptr[tag][word] > probability:
                best_tag, probability = tag, forward_ptr[tag][word]
        generated_pos_tags.append(best_tag)
    return generated_pos_tags

def write_to_output_file(lines, out_file, pos_tags, transition_probabilities, observation_likelihoods):
    with open(out_file, 'a') as output_file_handler:
        ctr = 0
        for line in lines[: -1]:
            forward_ptr = viterbi(transition_probabilities, observation_likelihoods, pos_tags, line)
            words = line.split(' ')
            word_tags = get_pos_tags(words, pos_tags, forward_ptr)
            new_line = ' '.join(['{}/{}'.format(words[i], word_tags[i]) for i in range(len(words))])
            output_file_handler.write(new_line + '\n')
            ctr += 1

def tag_sentence(test_file, model_file, out_file, start_time):
    lines = process_test_file(test_file)
    pos_tags, transition_probabilities, observation_likelihoods = process_model_file(model_file)
    pos_tags.remove('<s>')
    write_to_output_file(lines, out_file, pos_tags, transition_probabilities, observation_likelihoods)
    
if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file, start_time)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
