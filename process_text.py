import numpy as np
from store import DB


def get_all_messeges():
    db = DB()
    all_messages = db.cursor.execute('SELECT message FROM conversation_history').fetchall()
    return all_messages


def get_vocab(vocab_size=40, return_freq=False):
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

    freq = []
    for key in keys_freq[0:vocab_size]:
        freq.append(count[key])

    if return_freq:
        return vocab, freq

    else:
        return vocab


class Char2num():
    def __init__(self, vocab):
        self.vocab = vocab
        self.size_of_base_vocab = len(vocab) + 2  # add 2 beacuse of start and stop tokens
        self.start_token_num = len(vocab)
        self.stop_token_num = len(vocab) + 1
        self.char2nbr = self._setup_enc(vocab)
        self.nbr2char = {v: k for k, v in self.char2nbr.items()}

    def start_encoding(self):
        start_encoding = self.start_token_num
        return start_encoding

    def stop_encoding(self):
        stop_encoding = self.stop_token_num
        return stop_encoding

    @staticmethod
    def _setup_enc(vocab):
        char2nbr = dict()

        for (i, char) in enumerate(vocab):
            char2nbr[char] = i

        return char2nbr

    def remove_unknown(self, sentence):
        sentence = sentence.lower()
        filtered_sentence = ''
        for char in sentence:
            if char in self.vocab:
                filtered_sentence += char

        return filtered_sentence

    def num(self, sentence):
        sentence = self.remove_unknown(sentence)
        # Remove unknown tokens
        sentence_length = len(sentence)
        sentence_encoding = np.zeros((sentence_length))

        for (i, char) in enumerate(sentence):
            nbr = self.char2nbr[char]
            sentence_encoding[i] = nbr

        # add encodings for start and stop tokens
        start_enc = self.start_encoding()
        stop_enc = self.stop_encoding()
        sentence_encoding = np.concatenate(([start_enc], sentence_encoding, [stop_enc]))

        return sentence_encoding

    def sentence(self, encoding):
        # shape of encoding is (N, D) where N is length of sentence and D is size of num dim
        sentence = ''
        for char_num in encoding:

            if char_num in self.nbr2char.keys():  # this will make it ignore start and stop tokens
                char = self.nbr2char[char_num]
                sentence += char

        return sentence


if __name__ == '__main__':
    N = 50
    (vocab, freq) = get_vocab(N, return_freq=True)

    for j in range(N):
        print('%s | %d' % (vocab[j], freq[j]))

    char2num = Char2num(vocab)

    message = "art"
    encoding = char2num.num(message)
    sentence = char2num.sentence(encoding)
    assert(sentence == message)

    message = "art4"
    encoding = char2num.num(message)
    sentence = char2num.sentence(encoding)
    assert(sentence != message)
