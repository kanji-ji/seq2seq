import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import MeCab
import mojimoji
import pandas as pd
from sklearn.utils import shuffle
import torch


class DataBuilder(object):
    """read data from many formats. preprocess them. turn them into id np.array.
    """

    def __init__(self):
        """
        data_path (string): csv file path
        src_column, tgt_column (string)
        """
        self.data = pd.DataFrame(columns=['src', 'tgt'])
        self.data_size = len(self.data)
        self.lengths = None
        self.src_vocab = None
        self.tgt_vocab = None

    def add_data_from_csv(self,
                          data_path,
                          src_column,
                          tgt_column,
                          preprocess=True):
        """
        data_path (string): csv file path
        src_column, tgt_column (string)
        preprocess (bool) Default:True
        """
        data = pd.read_csv(data_path, encoding='utf-8')
        self.add_data(data, src_column, tgt_column, preprocess)

    def read_data_from_txt(self, data_path, is_src, preprocess=True):
        """
        data_path (string): txt file path
        is_src (bool): src text(True) or tgt text(False)
        preprocess (bool): preprocessing is necessary or not Default: True
        """
        column = 'src' if is_src else 'tgt'
        with open(data_path, 'r') as f:
            text = f.readlines()

        self.data[column] = pd.Series(text)

    def add_data(self, data, src_column, tgt_column, preprocess=True):
        data = data[[src_column, tgt_column]]
        data.columns = ['src', 'tgt']
        if preprocess:
            data = clean_tokenize(data)
        self.data = pd.concat([self.data, data], axis=0, ignore_index=True)
        self.data_size = len(self.data)

    def drop_long_seq(self, src_maxlen, tgt_maxlen):
        self.data = self.data[
            self.data['src'].str.split().apply(len) <= src_maxlen]
        self.data = self.data[
            self.data['tgt'].str.split().apply(len) <= tgt_maxlen]
        self.data.reset_index(drop=True, inplace=True)
        print('data size dropped from {} to {}({:.1f}% left)'.format(
            self.data_size, len(self.data),
            100 * len(self.data) / self.data_size))
        self.data_size = len(self.data)

    def build_vocab(self):
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()

        for i in range(self.data_size):
            for word in (self.data.loc[i, 'src']).split():
                self.src_vocab.add(word)
            for word in (self.data.loc[i, 'tgt']).split():
                self.tgt_vocab.add(word)

        return self.src_vocab, self.tgt_vocab

    def make_id_array(self, src_maxlen, tgt_maxlen):
        if self.src_vocab is None:
            raise Exception(
                'Vocab is not built. Please call build_vocab before.')

        self.drop_long_seq(src_maxlen, tgt_maxlen)

        src = np.zeros((self.data_size, src_maxlen), dtype='int32')
        tgt = np.zeros((self.data_size, tgt_maxlen), dtype='int32')

        for i in range(self.data_size):
            for j, word in enumerate(self.data.loc[i, 'src'].split()):
                src[i][j] = self.src_vocab.word2id(word)
            for j, word in enumerate(self.data.loc[i, 'tgt'].split()):
                tgt[i][j] = self.tgt_vocab.word2id(word)

        return src, tgt


class Vocab(object):
    """ useful dictionary that holds words and their ids.
    Attributes:
        min_count: to be implemented
        word2id_dict (dict):
        id2word_dict (dict):
        size (int)
    TODO:
        add min_count option
    """
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3
    num_id = 4
    alp_id = 5
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    num_token = '<NUM>'
    alp_token = '<ALP>'

    def __init__(self, min_count=0):
        self.word2id_dict = dict({
            Vocab.pad_token: Vocab.pad_id,
            Vocab.unk_token: Vocab.unk_id,
            Vocab.bos_token: Vocab.bos_id,
            Vocab.eos_token: Vocab.eos_id,
            Vocab.num_token: Vocab.num_id,
            Vocab.alp_token: Vocab.alp_id
        })
        self.id2word_dict = dict(
            {i: word
             for word, i in self.word2id_dict.items()})
        self.size = len(self.word2id_dict)
        self.min_count = min_count
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.size:
            self._i = 0
            raise StopIteration
        word = self.id2word(self._i)
        self._i += 1
        return word

    def add(self, word):
        """
        Args:
            word(string)
        if word is not in Vocab, then add it
        """
        key = self.word2id_dict.setdefault(word, self.size)
        self.id2word_dict[key] = word
        if key == self.size:
            self.size += 1

    def _delete(self, word):
        """
        Strongly unrecommended
        """
        try:
            key = self.word2id_dict.pop(word)
            self.id2word_dict.pop(key)
            self.size -= 1
        except KeyError:
            print('{} doesn\'t exist'.format(word))

    def word2id(self, word):
        """
        Args:
            word(string)
        Returns:
            returns id allocated to word if it's in Vocab. Otherwise, returns 1 which means unknown word.
        """
        return self.word2id_dict.get(word, Vocab.unk_id)

    def id2word(self, key):
        """
        Args:
            key(int)
        Returns:
            returns word allocated to key if it's in Vocab. Otherwise, returns <UNK>.
        """
        return self.id2word_dict.get(key, Vocab.unk_token)

    def build_vocab(self, sentences):
        """update vocab
        Args:
            sentences:list of lists,each element of list is one sentence,
            each sentence is represented as list of words
        """
        assert isinstance(sentences, list)

        for sentence in sentences:
            assert isinstance(sentence, list)
            for word in sentence:
                self.add(word)

    def seq2ids(self, sentence):
        """
        Args:
            sequence: list each element of which is word(string)
        Returns:
            list each element of which is id(int) corresponding to each word
        """
        assert isinstance(sentence, list)
        id_seq = list()
        for word in sentence:
            id_seq.append(self.word2id(word))

        return id_seq

    def ids2seq(self, id_seq):
        """inverse processing of seq2ids
        """
        assert isinstance(id_seq, list)
        sentence = list()
        for key in id_seq:
            sentence.append(self.id2word(key))
            if sentence[-1] == Vocab.eos_token:
                break
        return sentence


class DataLoader(object):
    """Data loader to return minibatches of input sequence and target sequence an iteration
    Attributes:
        input_seq: input sequence, numpy ndarray
        target_seq: target sequence, numpy ndarray
        input_lengths: true lengths of input sequences, before padding
        batch_size: batch size
    """

    def __init__(self, src_seq, tgt_seq, src_lengths, batch_size,
                 shuffle=True):
        self.src_seq = src_seq
        self.tgt_seq = tgt_seq
        self.src_lengths = src_lengths
        self.batch_size = batch_size
        self.size = len(self.src_seq)
        self.start_index = 0
        self.shuffle = shuffle
        if self.shuffle:
            self.reset()

    def reset(self):
        """shuffle data
        """
        self.src_seq, self.tgt_seq, self.src_lengths = shuffle(
            self.src_seq, self.tgt_seq, self.src_lengths)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_index >= self.size:
            if self.shuffle:
                self.reset()
            self.start_index = 0
            raise StopIteration
        batch_X = self.src_seq[self.start_index:self.start_index +
                               self.batch_size]
        batch_Y = self.tgt_seq[self.start_index:self.start_index +
                               self.batch_size]
        lengths = self.src_lengths[self.start_index:self.start_index +
                                   self.batch_size]
        self.start_index += self.batch_size

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)

        lengths, perm_idx = lengths.sort(descending=True)
        batch_X = batch_X[perm_idx]
        batch_Y = batch_Y[perm_idx]
        return batch_X, batch_Y, lengths


def remove_symbol(text):
    """入力されたテキストから句読点などの不要な記号をいくつか削除する。
    """
    remove_list = [
        ',', '.', '-', '、', '，', '。', '\ufeff', '\u3000', '「', '」', '（', '）',
        '(', ')', '\n'
    ]
    for i, symbol in enumerate(remove_list):
        text = text.replace(symbol, '')
    return text


def add_bos_eos(text):
    return Vocab.bos_token + ' ' + text + ' ' + Vocab.eos_token


def replace_number(text):
    new_text = ''
    for word in text.split():
        if word.isdigit():
            new_text += Vocab.num_token + ' '
        elif word == Vocab.eos_token:
            new_text += Vocab.eos_token
        else:
            new_text += word + ' '
    return new_text


def isalpha(s):
    """
    Args:
        s:string
    Returns:
        bool:sが半角英字から成るかどうか
    """
    alphaReg = re.compile(r'^[a-zA-Z]+$')
    return alphaReg.match(s) is not None


def replace_alphabet(text):
    new_text = ''
    for word in text.split():
        if isalpha(word):
            new_text += Vocab.alp_token + ' '
        elif word == Vocab.eos_token:
            new_text += word
        else:
            new_text += word + ' '
    return new_text


def replace_unknown(text, unknown_set):
    new_text = ''
    for word in text.split():
        if word in unknown_set:
            new_text += Vocab.unk_token + ' '
        elif word == Vocab.eos_token:
            new_text += word
        else:
            new_text += word + ' '
    return new_text


def clean_tokenize(data):
    """
    data (pandas.DataFrmae):
    """

    def zen_to_han(text):
        text = mojimoji.zen_to_han(text, kana=False)
        return text

    m = MeCab.Tagger('-Owakati')
    data = data.applymap(zen_to_han)
    data = data.applymap(remove_symbol)
    data = data.applymap(m.parse)
    data = data.applymap(add_bos_eos)
    data = data.applymap(replace_number)
    data = data.applymap(replace_alphabet)
    return data


def calc_bleu(y_pred, y_true):
    """
    src(list of lists):
    tgt(list of lists):
    """

    bleu = 0.0
    cc = SmoothingFunction()

    for hyp, ref in zip(y_pred, y_true):
        bleu += sentence_bleu([ref], hyp, smoothing_function=cc.method1)

    bleu /= len(y_pred)
    bleu *= 100

    return bleu


def get_embedding_matrix(vocab, word2vec):

    embedding_matrix = np.random.uniform(
        low=-0.05, high=0.05, size=(src_words.size, word2vec.size))
    unknown_set = set()

    for i, word in enumerate(vocab):
        try:
            embedding_matrix[i] = word2vec[word]
        except KeyError:
            if word not in unknown_set:
                unknown_set.add(word)

    embedding_matrix[0] = np.zeros((word2vec.size, ))

    embedding_matrix = embedding_matrix.astype('float32')

    unknown_set.remove(Vocab.pad_token)
    unknown_set.remove(Vocab.bos_token)
    unknown_set.remove(Vocab.eos_token)
    unknown_set.remove(Vocab.unk_token)
    unknown_set.remove(Vocab.num_token)
    unknown_set.remove(Vocab.alp_token)

    return embedding_matrix, unknown_set
