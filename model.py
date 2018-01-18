import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import process_text
import matplotlib.pyplot as plt


class TextGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, start_token, stop_token):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.start_token = start_token
        self.stop_token = stop_token

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, encoded_sentence):
        lstm_out, self.hidden = \
            self.lstm(encoded_sentence.view(len(encoded_sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(encoded_sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def sample(self, argmax=False):
        self.hidden = self.init_hidden()

        encoded_sentence = [self.start_token]
        inputs = autograd.Variable(torch.from_numpy(np.array([self.start_token]))).float()

        while True:
            lstm_out, self.hidden = \
                self.lstm(inputs.view(len(inputs), 1, -1), self.hidden)
            tag_space = self.hidden2tag(lstm_out.view(len(inputs), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)

            if argmax:
                _, idx = torch.max(tag_scores, dim=1)
            else:
                idx = torch.multinomial(torch.exp(tag_scores), 1, replacement=True)
            encoded_sentence.append(int(idx.data))
            inputs = idx.float()

            if int(idx.data) == self.stop_token or len(encoded_sentence) > 100:
                return encoded_sentence


if __name__ == '__main__':
    ALL_MESSAGES = process_text.get_all_messeges()
    NUM_EPOCHS = 5000

#    ALL_MESSAGES = [("bil", ), ("flyg", )]

    ONEHOT_DIM = 1
    HIDDEN_DIM = 512

    TARGET_SIZE = 40

    vocab = process_text.get_vocab(TARGET_SIZE - 2)

    start_token = TARGET_SIZE - 2
    stop_token = TARGET_SIZE - 1
    char2num = process_text.Char2num(vocab)

    model = TextGenerator(ONEHOT_DIM, HIDDEN_DIM, TARGET_SIZE, start_token, stop_token)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # This will be the train loop
    historic_loss = []
    historic_step = []
    step = 0
    plt.figure()
    for ep in range(NUM_EPOCHS):
        random.shuffle(ALL_MESSAGES)
        plt.clf()
        print('------ epoch: %d ------' % ep)
        for (msg, ) in ALL_MESSAGES:
            encoding = char2num.num(msg)
            sentence_in = encoding[0:-1]  # include start token, exclude stop token
            targets = encoding[1:]  # exclude start token, include stop token

            sentence_in = autograd.Variable(torch.from_numpy(sentence_in)).float()
            targets = autograd.Variable(torch.from_numpy(targets)).long()

            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model(sentence_in)

            vals, idx = torch.max(tag_scores, dim=1)
            loss = loss_function(tag_scores, targets)

            (ll, ) = loss.data.numpy()
            historic_loss.append(ll)
            historic_step.append(step)

            if (step % 100) == 0:
                print('%d / %d' % (step % len(ALL_MESSAGES), len(ALL_MESSAGES)))

            step += 1
            loss.backward()
            optimizer.step()

        print('\n\n\nSample from LSTM:')
        sample = model.sample()
        print(char2num.sentence(sample))
        print('\nArgMax Sample from LSTM:')
        sample = model.sample(argmax=True)
        print(char2num.sentence(sample))
        print('\n\n\n')

        plt.plot(historic_step, historic_loss)
        plt.grid('on')
        plt.pause(0.125)
