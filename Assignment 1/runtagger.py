# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import numpy as np
import os
import math
import sys
import datetime

def viterbi(A, B, N, T):
    matrix_tensor = np.zeros([N + 2, T])
    for s in range(1, N + 1):
        matrix[s, 1] = A[0][s] * B[0][s]
    
    for t in range(2, T + 1):
        for s in range(1, N + 1):
            matrix[s, t] = max([matrix[s0][t - 1] * A[s0][s] * B[t][s0] for s0 in range(N)])

    matrix[N + 1][T] = max([matrix[s0][T] * A[s0][N + 1] for s0 in range(N)])
    return matrix

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    print(viterbi(A, B, N, T))

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
