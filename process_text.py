import numpy as np
from store import DB


def get_vocab(vocab_size=40):
    db = DB()
    all_messages = db.cursor.execute('SELECT message FROM conversation_history').fetchall()

    count = dict()
    for msg in all_messages:
        (m, ) = msg
        ml = m.lower()

        for m in ml:
            if m in count.keys():
                count[m] += 1
            else:
                count[m] = 1

    keys_freq = sorted(count, key=count.get, reverse=True)

    vocab = []
    for i in range(vocab_size):
        vocab.append(keys_freq[i])

    return vocab


class Char2onehot():
    def __init__(self, vocab):
        self.size_of_base_vocab = len(vocab) + 2  # add 2 beacuse of start and stop tokens
        self.start_token_idx = len(vocab)
        self.stop_token_idx = len(vocab) + 1
        self.char2nbr = self._setup_enc(vocab)
        self.nbr2char = {v: k for k, v in self.char2nbr.items()}

    def start_encoding(self):
        start_encoding = np.zeros((self.size_of_base_vocab))
        start_encoding[self.start_token_idx] = 1
        return start_encoding

    def stop_encoding(self):
        stop_encoding = np.zeros((self.size_of_base_vocab))
        stop_encoding[self.stop_token_idx] = 1
        return stop_encoding

    @staticmethod
    def _setup_enc(vocab):
        char2nbr = dict()

        for (i, char) in enumerate(vocab):
            char2nbr[char] = i

        return char2nbr

    def onehot(self, sentence):
        sentence_length = len(sentence)
        sentence_encoding = np.zeros((sentence_length, self.size_of_base_vocab))

        for (i, char) in enumerate(sentence):
            print(char)
            idx = self.char2nbr[char]
            sentence_encoding[i][idx] = 1

        # add encodings for start and stop tokens
        start_enc = self.start_encoding()
        stop_enc = self.stop_encoding()

        sentence_encoding = np.concatenate(
            (start_enc[None, :], sentence_encoding, stop_enc[None, :]))

        return sentence_encoding

    def sentence(self, encoding):
        # shape of encoding is (N, D) where N is length of sentence and D is size of onehot dim
        sentence = ''
        for char_encoding in encoding[1:-1, :]:  # ignore start and stop tokens
            ((nbr, ), ) = np.where(char_encoding == 1)
            char = self.nbr2char[nbr]
            sentence += char

        return sentence


vocab = get_vocab()
print(vocab)
char2onehot = Char2onehot(vocab)

encoding = char2onehot.onehot("hej")
sentence = char2onehot.sentence(encoding)
print(sentence)
