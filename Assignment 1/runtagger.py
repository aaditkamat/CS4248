import datetime
import pickle
import sys
from collections import defaultdict

START_TOKEN = "<s>"
END_TOKEN = "</s>"


def smoothen(unigram_counts, bigram_counts):
    bigram_probs = {}
    for bigram, bigram_count in bigram_count.items():
        bigram_probs[bigram] = bigram_count / unigram_counts[bigram[0]]
    return bigram_probs


def preprocess(lines, model):
    word_counts = model["word_counts"]
    bigram_counts = model["bigram_counts"]
    emission_probs = model["emission_probs"]
    model["emission_probs"] = smoothen(word_counts, bigram_counts, emission_probs)
    tag_counts = model["word_counts"]
    model["transition_probs"] = smoothen(tag_counts, bigram_counts)


def viterbi(model, line):
    words = line.split(" ")
    transition_probs = model["transition_probs"]
    emission_probs = model["emission_probs"]
    tags = model["tag_counts"].keys()
    viterbi_vals = defaultdict(float)

    for tag in tags:
        viterbi_vals[(tag, words[0])] = (
            transition_probs[(START_TOKEN, tag)] * emission_probs[(tag, words[0])]
        )

    for i in range(1, len(words)):
        word1, word2 = words[i - 1 : i + 1]
        for tag2 in tags:
            viterbi_vals[(tag2, word2)] = max(
                [
                    viterbi_vals[(tag1, word1)]
                    * transition_probs[(tag1, tag2)]
                    * emission_probs[(tag2, word2)]
                    for tag1 in tags
                ]
            )

    viterbi_vals[(END_TOKEN, words[-1])] = max(
        [
            viterbi_vals[(tag, words[-1])] * transition_probs[(tag, END_TOKEN)]
            for tag in tags
        ]
    )

    return viterbi_vals


def read_test_file(test_file):
    with open(test_file) as test_file_handler:
        lines = test_file_handler.read().split("\n")
        return lines


# Objects obtained from the pickled file need to be converted to the
# custom defaultdict type because they are stored as dictionaries


def restore_types(model):
    typed_model = {}
    for name, dictionary in model.items():
        default_factory = type(list(dictionary.values())[0])
        if default_factory == int:

            def default_factory():
                return 1

        new_dictionary = defaultdict(default_factory)
        for key in dictionary:
            new_dictionary[key] = dictionary[key]
        typed_model[name] = new_dictionary
    return typed_model


def read_model_file(model_file):
    with open(model_file, mode="rb") as model_file_handler:
        model = pickle.load(model_file_handler)
        return restore_types(model)


def get_pos_tags(words, tags, viterbi_vals):
    generated_pos_tags = []
    for word in words:
        best_tag, prob = None, 0
        for tag in tags:
            if viterbi_vals[(tag, word)] >= prob:
                best_tag, prob = tag, viterbi_vals[(tag, word)]
        generated_pos_tags.append(best_tag)
    return generated_pos_tags


def write_to_output_file(lines, out_file, model):
    with open(out_file, "w") as output_file_handler:
        ctr = 0
        for line in lines[:-1]:
            viterbi_vals = viterbi(model, line)
            words = line.split(" ")

            tags = model["tag_counts"].keys()

            word_tags = get_pos_tags(words, tags, viterbi_vals)
            new_line = " ".join(
                ["{}/{}".format(words[i], word_tags[i]) for i in range(len(words))]
            )
            output_file_handler.write(new_line + "\n")
            ctr += 1


def tag_sentence(test_file, model_file, out_file, start_time):
    lines = read_test_file(test_file)
    model = read_model_file(model_file)
    write_to_output_file(lines, out_file, model)


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file, start_time)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
