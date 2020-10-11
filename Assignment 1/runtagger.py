# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import pdb
import numpy as np
import os
import json
import math
import sys
import datetime

def viterbi(A, B, N, T):
    matrix  = np.zeros([N, T])
    for s in range(N):
        matrix[s, 0] = A[0][s] * B[s][0]
    
    for t in range(T):
        for s in range(N):
            matrix[s, t] = max([matrix[s0][t] * A[s0][s] * B[s][t] for s0 in range(N)])

    return matrix

def get_words(lines):
    words = []
    for line in lines:
        words.extend(line.split(' '))
    return words

def process_test_file(test_file):
    with open(test_file) as test_file_handler:
        lines = test_file_handler.read().split('\n')
        return lines

def process_model_file(model_file):
    with open(model_file) as model_file_handler:
        model = json.load(model_file_handler)
        return model["pos_tags"], np.array(model["transition_probabilities"]), np.array(model["observation_likelihoods"])

def get_pos_tag(line, index, viterbi_matrix, pos_tags):
    column = list(viterbi_matrix[:, index])
    print(column)
    return pos_tags[column.index(max(column))]

def write_to_output_file(word_pos_tag_mapping, lines, out_file):
    for line in lines:
        new_line = ' '.join(['{}/{}'.format(word, word_pos_tag_mapping[word]) for word in word_pos_tag_mapping])
        with open(out_file, mode='a') as output_file_handler:
            output_file_handler.write(new_line + '\n')

def tag_sentence(test_file, model_file, out_file, start_time):
    lines = process_test_file(test_file)
    words = get_words(lines)
    pos_tags, A, B = process_model_file(model_file)
    no_of_observations = B.shape[1]
    word_pos_tag_mapping = {}
    for start in range(0, 1000, no_of_observations):
        observations = words[start: start + no_of_observations]
        no_of_observations = len(observations)
        viterbi_matrix = viterbi(A, B, len(pos_tags), len(observations))
        for index in range(no_of_observations):
            pos_tag = get_pos_tag(observations, index, viterbi_matrix, pos_tags)
            word_pos_tag_mapping[observations[index]] = pos_tag
    pdb.set_trace()
    write_to_output_file(word_pos_tag_mapping, lines, out_file)

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file, start_time)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
