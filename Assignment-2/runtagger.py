import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

with open("sents.test", mode="r") as test_file_handler:
    test_data = []
    for line in test_file_handler.read().split("\n"):
        sentence = line.lower().split()
        test_data.append(sentence)
    test_data = test_data[:-1]


def get_ix(seq, to_ix, start_index):
    for token in seq:
        if token not in to_ix:
            to_ix[token] = len(to_ix) + start_index


test_words_to_ix = {}
for sentence in test_data:
    get_ix(sentence, test_words_to_ix, 2)
test_words_to_ix["-PAD-"] = 0
test_words_to_ix["-OOV-"] = 1


class BiLSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, padding_idx):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence_batch, dropout_rate=0.25):
        embeds = self.word_embeddings(sentence_batch)

        lstm_out, _ = self.lstm(embeds)
        lstm_dropout = F.dropout(lstm_out, dropout_rate)

        tag_space = self.hidden2tag(lstm_dropout)

        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


meta_data = torch.load("sent-model")
(
    EMBEDDING_DIM,
    HIDDEN_DIM,
    BATCH_SIZE,
    training_words_to_ix,
    tags_to_ix,
    max_length,
) = meta_data["hyperparameters"]
model = BiLSTMTagger(
    EMBEDDING_DIM, HIDDEN_DIM, len(training_words_to_ix), len(tags_to_ix), 0
)
model.load_state_dict(meta_data["model"])


def create_test_batches(permutations, test_data):
    sentences = []
    for index in permutations[i : i + BATCH_SIZE]:
        sentences.append([word for word in test_data[index]])
    return sentences


# Size of sequences is: [sentence_length, batch_size]
def prepare_sequences(seqs, to_ix, max_length):
    padded_tensors = []
    for seq in seqs:
        idxs = [to_ix[token] if token.lower() in to_ix else 0 for token in seq]
        orig_tensor = torch.tensor(idxs, dtype=torch.long)
        padded_tensor = F.pad(orig_tensor, (0, max_length - len(orig_tensor))).view(
            1, max_length
        )
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors).transpose(0, 1)


def get_tag(tags_to_ix, curr_idx):
    return [tag for tag, idx in tags_to_ix.items() if idx == curr_idx][0]


def generate_tags(tag_scores, tags_to_ix):
    idxs = torch.max(tag_scores, 1)[1]
    return [get_tag(tags_to_ix, idx) for idx in idxs]


with open("sents.out", "w") as output_file_handler:
    indices = list(range(len(test_data)))
    for i in range(0, len(test_data), BATCH_SIZE):
        sentences = create_test_batches(indices, test_data)
        if len(sentences) == 0:
            print("No more sentences")
            break
        inputs = prepare_sequences(sentences, test_words_to_ix, max_length)
        tag_scores = model(inputs)
        tag_scores = tag_scores.view(-1, tag_scores.shape[-1])

        tags = generate_tags(tag_scores, tags_to_ix)
        assert inputs.shape[0] * inputs.shape[1] == len(tags)

        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                output_file_handler.write(
                    "{}/{} ".format(sentences[i][j], tags[i * len(sentences[i]) + j])
                )
            output_file_handler.write("\n")
