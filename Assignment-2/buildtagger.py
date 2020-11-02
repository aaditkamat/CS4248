import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

with open("sents.train", mode="r") as train_file_handler:
    training_data = []
    for line in train_file_handler.read().split("\n"):
        tokens = line.split()
        sentence = ["/".join(token.split("/")[:-1]).lower() for token in tokens]
        tags = [token.split("/")[-1] for token in tokens]
        training_data.append((sentence, tags))
    training_data = training_data[:-1]


def get_ix(seq, to_ix, start_index):
    for token in seq:
        if token not in to_ix:
            to_ix[token] = len(to_ix) + start_index


words_to_ix, tags_to_ix = {}, {}
for sentence, tags in training_data:
    get_ix(sentence, words_to_ix, 2)
    get_ix(tags, tags_to_ix, 1)
words_to_ix["-PAD-"] = 0
words_to_ix["-OOV-"] = 1
tags_to_ix["-PAD-"] = 0

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


EMBEDDING_DIM = 64
HIDDEN_DIM = 64

model = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(words_to_ix), len(tags_to_ix), 0)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
max_length = max([len(sentence) for sentence, tags in training_data])


def create_train_batches(permutations, training_data):
    sentences, tags = [], []
    for index in permutations[i : i + BATCH_SIZE]:
        sentences.append([word.lower() for word in training_data[index][0]])
        tags.append(training_data[index][1])
    return sentences, tags


BATCH_SIZE = 128
EPOCHS = 2

for _ in range(EPOCHS):
    permutations = torch.randperm(len(training_data))
    start_time = datetime.datetime.now()
    for i in range(0, len(training_data), BATCH_SIZE):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Split training data into minibatches
        sentences, tags = create_train_batches(permutations, training_data)

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentences_in = prepare_sequences(sentences, words_to_ix, max_length)
        targets = prepare_sequences(tags, tags_to_ix, max_length)

        # Step 3. Run our forward pass.
        tag_scores = model(sentences_in)
        tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
        targets = targets.reshape(-1)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        if i % (BATCH_SIZE * 100) == 0 and i >= (BATCH_SIZE * 100):
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).seconds
            elapsed_mins = elapsed_time // 60
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
            print(
                "100 more batches processed in {}m and {}s".format(
                    elapsed_mins, elapsed_secs
                )
            )
            print("Loss: {}".format(loss))
            start_time = datetime.datetime.now()

torch.save(
    {
        "model": model.state_dict(),
        "hyperparameters": [
            EMBEDDING_DIM,
            HIDDEN_DIM,
            BATCH_SIZE,
            words_to_ix,
            tags_to_ix,
            max_length,
        ],
    },
    "sent-model",
)
