import datetime
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
WORD_PAD_IX = 1
TAG_PAD_IX = 0
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
DROPOUT_RATE = 0.25
BATCH_SIZE = 128


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

    def forward(self, sentence_batch):
        embeds = F.dropout(
            input=self.word_embeddings(sentence_batch), p=self.dropout_rate
        )

        lstm_out, _ = self.lstm(embeds)

        tag_scores = self.hidden2tag(
            F.dropout(input=lstm_out, p=self.dropout_rate)
        )

        return tag_scores


def get_ix(seq, to_ix, start_index):
    for token in seq:
        if token not in to_ix:
            to_ix[token] = len(to_ix) + start_index


def to_ix(training_data):
    words_to_ix, tags_to_ix = {}, {}
    for sentence, tags in training_data:
        get_ix(sentence, words_to_ix, WORD_PAD_IX + 1)
        get_ix(tags, tags_to_ix, TAG_PAD_IX + 1)
    words_to_ix["-UNK-"] = WORD_PAD_IX - 1
    words_to_ix["-PAD-"] = WORD_PAD_IX
    tags_to_ix["-PAD-"] = TAG_PAD_IX
    return words_to_ix, tags_to_ix


def prepare_sequence(seq, to_ix, pad_ix, pad_length):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    idxs.extend([pad_ix for _ in range(pad_length - len(idxs))])
    return idxs


def form_batch(data, size):
    new_data = []
    for i in range(0, len(data), size):
        new_data.append(data[i : i + size])
    return new_data


def create_batches(training_data, words_to_ix, tags_to_ix):
    batched_data = form_batch(training_data, BATCH_SIZE)

    sentences_data, tags_data = [], []
    for batch in batched_data:
        sentence_batch, tag_batch = [], []
        max_sentences_length = max([len(sentences) for sentences, _ in batch])
        max_tags_length = max([len(tags) for _, tags in batch])
        for sentences, tags in batch:
            numericalized_sentences = prepare_sequence(
                sentences, words_to_ix, WORD_PAD_IX, max_sentences_length
            )
            numericalized_tags = prepare_sequence(
                tags, tags_to_ix, TAG_PAD_IX, max_tags_length
            )
            sentence_batch.append(numericalized_sentences)
            tag_batch.append(numericalized_tags)
        sentence_batch = torch.LongTensor(sentence_batch).t().to(DEVICE)
        tag_batch = torch.LongTensor(tag_batch).t().to(DEVICE)
        sentences_data.append(sentence_batch)
        tags_data.append(tag_batch)
    return sentences_data, tags_data


def perform_training(sentences_data, tags_data, words_to_ix, tags_to_ix):
    model = BiLSTMTagger(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        len(words_to_ix),
        len(tags_to_ix),
        WORD_PAD_IX,
        DROPOUT_RATE,
    ).to(DEVICE)

    loss_function = nn.NLLLoss(ignore_index=TAG_PAD_IX).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(EPOCHS):
        for j in range(len(sentences_data)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()

            # Step 2. Run our forward pass.
            tag_scores = model(sentences_data[j])
            tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
            targets = tags_data[j].reshape(-1)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    return model


def parse(train_file):
    with open(train_file, mode="r") as train_file_handler:
        training_data = []

        for line in train_file_handler.read().split("\n"):
            tokens = line.split()
            sentence = [
                "/".join(token.split("/")[:-1]).lower() for token in tokens
            ]
            tags = [token.split("/")[-1] for token in tokens]
            training_data.append((sentence, tags))

        training_data = training_data[:-1]

    return training_data


def train_model(train_file, model_file):
    training_data = parse(train_file)
    words_to_ix, tags_to_ix = to_ix(training_data)
    sentences_data, tags_data = create_batches(
        training_data, words_to_ix, tags_to_ix
    )
    model = perform_training(sentences_data, tags_data, words_to_ix, tags_to_ix)
    torch.save(
        {
            "model": model.state_dict(),
            "hyperparameters": [
                EMBEDDING_DIM,
                HIDDEN_DIM,
                BATCH_SIZE,
                WORD_PAD_IX,
                DROPOUT_RATE,
                words_to_ix,
                tags_to_ix,
            ],
        },
        model_file,
    )


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
