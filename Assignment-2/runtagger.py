import datetime
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn._functions import padding

from buildtagger import BATCH_SIZE, WORD_PAD_IX, perform_training

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ix(seq, to_ix, start_index):
    for token in seq:
        if token not in to_ix:
            to_ix[token] = len(to_ix) + start_index


def to_ix(test_data):
    words_to_ix = {}
    for sentence in test_data:
        get_ix(sentence, words_to_ix, 2)
    words_to_ix["-PAD-"] = 0
    words_to_ix["-OOV-"] = 1
    return words_to_ix


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        tagset_size,
        padding_idx,
        dropout_rate,
    ):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout_rate = dropout_rate
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate,
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence_batch):
        embeds = self.word_embeddings(sentence_batch)

        lstm_out, _ = self.lstm(embeds)
        lstm_dropout = F.dropout(lstm_out, self.dropout_rate)

        tag_space = F.dropout(self.hidden2tag(lstm_dropout), self.dropout_rate)

        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


def get_model_parameters(model_file):
    meta_data = torch.load(model_file)
    (
        EMBEDDING_DIM,
        HIDDEN_DIM,
        BATCH_SIZE,
        WORD_PAD_IX,
        DROPOUT_RATE,
        words_to_ix,
        tags_to_ix,
    ) = meta_data["hyperparameters"]
    model = BiLSTMTagger(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        len(words_to_ix),
        len(tags_to_ix),
        WORD_PAD_IX,
        DROPOUT_RATE,
    )
    model.load_state_dict(meta_data["model"])
    return model, tags_to_ix, WORD_PAD_IX, BATCH_SIZE


def prepare_sequence(seq, to_ix, pad_ix, pad_length):
    idxs = [to_ix[w] for w in seq]
    idxs.extend([pad_ix for _ in range(pad_length - len(idxs))])
    return idxs


def form_batch(data, size):
    new_data = []
    for i in range(0, len(data), size):
        new_data.append(data[i : i + size])
    return new_data


def create_batches(test_data, words_to_ix, padding_ix, batch_size):
    batched_data = form_batch(test_data, batch_size)

    sentences_data = []
    for batch in batched_data:
        sentence_batch = []
        max_sentences_length = max([len(sentences) for sentences in batch])
        for sentences in batch:
            numericalized_sentences = prepare_sequence(
                sentences, words_to_ix, padding_ix, max_sentences_length
            )
            sentence_batch.append(numericalized_sentences)
            sentence_batch = torch.tensor(sentence_batch, device=DEVICE)
            sentences_data.append(sentence_batch, device=DEVICE)

    return sentences_data


def get_tag(tags_to_ix, curr_idx):
    return [tag for tag, idx in tags_to_ix.items() if idx == curr_idx][0]


def generate_tags(tag_scores, tags_to_ix):
    idxs = torch.max(tag_scores, 1)[1]
    return [get_tag(tags_to_ix, idx) for idx in idxs]


def perform_tagging(
    out_file, test_data, model, words_to_ix, tags_to_ix, sentences_data
):
    with open(out_file, "w") as output_file_handler:
        all_tags = []
        for i in range(len(sentences_data)):
            tag_scores = model(sentences_data[i])
            tag_scores = tag_scores.view(-1, tag_scores.shape[-1])

            tags = generate_tags(tag_scores, tags_to_ix)
            all_tags.extend(tags)

        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                output_file_handler.write(
                    "{}/{} ".format(
                        test_data[i][j], all_tags[i * len(test_data[0]) + j]
                    )
                )
            output_file_handler.write("\n")


def parse(test_file):
    with open(test_file, mode="r") as test_file_handler:
        test_data = []
        for line in test_file_handler.read().split("\n"):
            sentence = line.lower().split()
            test_data.append(sentence)
        test_data = test_data[:-1]
    return test_data


def tag_sentence(test_file, model_file, out_file):
    test_data = parse(test_file)
    model, tags_to_ix, WORD_PAD_IX, BATCH_SIZE = get_model_parameters(model_file)
    words_to_ix = to_ix(test_data)
    sentences_data = create_batches(test_data, words_to_ix, WORD_PAD_IX, BATCH_SIZE)
    perform_tagging(out_file, test_data, model, words_to_ix, tags_to_ix, sentences_data)


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
