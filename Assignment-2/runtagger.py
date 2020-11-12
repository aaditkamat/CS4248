import datetime
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ix(seq, to_ix, start_index):
    for token in seq:
        if token not in to_ix:
            to_ix[token] = len(to_ix) + start_index


def to_ix(test_data, pad_ix):
    words_to_ix = {}
    for sentence in test_data:
        get_ix(sentence, words_to_ix, pad_ix + 1)
    words_to_ix["-UNK-"] = pad_ix - 1
    words_to_ix["-PAD-"] = pad_ix
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

        self.dropout_rate = dropout_rate

    def forward(self, sentence):
        embeds = F.dropout(
            input=self.word_embeddings(sentence), p=self.dropout_rate
        )

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        tag_scores = self.hidden2tag(
            F.dropout(
                input=lstm_out.view(len(sentence), -1), p=self.dropout_rate
            )
        )

        return tag_scores


def prepare_sequence(seq, to_ix, pad_ix):
    idxs = [to_ix[w] if w in to_ix else pad_ix for w in seq]
    return torch.LongTensor(idxs).to(DEVICE)


def get_tag(tags_to_ix, curr_idx):
    return [tag for tag, idx in tags_to_ix.items() if idx == curr_idx][0]


def generate_tags(tag_scores, tags_to_ix):
    idxs = torch.max(tag_scores, 1)[1]
    return [get_tag(tags_to_ix, idx) for idx in idxs]


def perform_tagging(
    out_file, test_data, model, word_pad_ix, words_to_ix, tags_to_ix
):
    with open(out_file, "w") as output_file_handler:
        for i in range(len(test_data)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            inputs = prepare_sequence(test_data[i], words_to_ix, word_pad_ix)
            tag_scores = model(inputs)
            tag_scores = tag_scores.view(-1, tag_scores.shape[-1])

            tags = generate_tags(tag_scores, tags_to_ix)
            assert len(test_data[i]) == len(tags)
            for j in range(len(test_data[i])):
                output_file_handler.write(
                    "{}/{} ".format(test_data[i][j], tags[j])
                )
            output_file_handler.write("\n")

            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))


def parse(test_file):
    with open(test_file, mode="r") as test_file_handler:
        test_data = []
        for line in test_file_handler.read().split("\n"):
            sentence = line.split()
            test_data.append(sentence)
        test_data = test_data[:-1]
    return test_data


def tag_sentence(test_file, model_file, out_file):
    test_data = parse(test_file)

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
    model.to(DEVICE)

    perform_tagging(
        out_file, test_data, model, WORD_PAD_IX, words_to_ix, tags_to_ix
    )


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
