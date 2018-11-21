import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
from models import seq2seq
import utils
from utils import *

MODEL_PATH = "./saved_models/"
SAMPLE_PATH = './samples/'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument(
        '--src_maxlen',
        type=int,
        help='this should be equal to its counterpart in training')
    parser.add_argument(
        '--tgt_maxlen',
        type=int,
        help='this should be equal to its counterpart in training')
    parser.add_argument(
        '--model_file',
        type=str,
        help='this should be equal to its counterpart in training')
    parser.add_argument('--data_path', type=str, help='test data')
    parser.add_argument(
        '--sample_file',
        type=str,
        help='csv file to write inference results in')
    parser.add_argument('--attention')

    args = parser.parse_args()
    batch_size = args.batch_size if args.batch_size is not None else 20
    src_maxlen = args.src_maxlen if args.src_maxlen is not None else 100
    tgt_maxlen = args.tgt_maxlen if args.tgt_maxlen is not None else 100
    model_file = args.model_file if args.model_file is not None else 'tmp.model'
    data_path = args.data_path if args.data_path is not None else './data/test.csv'
    sample_file = args.sample_file if args.sample_file is not None else 'tmp.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = utils.DataReader()
    test.add_data_from_csv(data_path, 'src', 'tgt', preprocess=False)

    test.drop_long_seq(src_maxlen, tgt_maxlen)

    src_lengths = test.data['src'].str.split().apply(len)
    src_lengths = np.array(src_lengths).astype('int32') - 1  #後で<EOS>を削除するため

    data_size = test.data_size
    src = np.zeros((data_size, src_maxlen), dtype='int32')
    tgt = np.zeros((data_size, tgt_maxlen), dtype='int32')

    with open('src.vocab', 'rb') as f:
        src_words = pickle.load(f)

    with open('tgt.vocab', 'rb') as f:
        tgt_words = pickle.load(f)

    with open('unknown.json', 'r') as f:
        unknown_list = json.loads(f.read(), 'utf-8')
        unknwon_set = set(unknown_list)

    for i in range(data_size):
        for j, word in enumerate(test.data.loc[i, 'src'].split()):
            if word in unknown_set:
                word = utils.Vocab.unk_token
            src[i][j] = src_words.word2id(word)
        for j, word in enumerate(test.data.loc[i, 'tgt'].split()):
            if word in unknown_set:
                word = utils.Vocab.unk_token
            tgt[i][j] = tgt_words.word2id(word)

    src = src[:, :-1]  #<EOS>削除
    tgt = tgt[:, 1:]  #<BOS>削除

    test_dataloader = DataLoader(src, tgt, src_lengths, batch_size=batch_size)

    with open('params.json', 'r') as f:
        params = json.load(f)
    assert isinstance(params, dict)

    #TODO: モデルを引数によって選べるようにする
    model = seq2seq.EncoderDecoder(**params).to(device)
    print('Loading model...')
    model.load_state_dict(torch.load(MODEL_PATH + model_file))

    iteration = 100 // batch_size

    with open(SAMPLE_PATH + sample_file, 'w') as f:

        for i in range(iteration):

            test_dataloader.reset()

            batch_X, batch_Y, X_length = next(test_dataloader)

            tgt_length = batch_Y.size(1)

            y_pred = model.sample(batch_X, X_length, tgt_length, tgt_words)

            X = batch_X.tolist()
            Y_true = batch_Y.tolist()
            Y_pred = y_pred.tolist()

            for x, y_true, y_pred in zip(X, Y_true, Y_pred):
                x = src_words.ids2seq(x)
                y_true = tgt_words.ids2seq(y_true)
                y_pred = tgt_words.ids2seq(y_pred)
                x = ' '.join(x)
                y_true = ' '.join(y_true)
                y_pred = ' '.join(y_pred)
                #print(x)
                #print(y_true)
                print(y_pred)
                f.write(x + ',' + y_true + ',' + y_pred + '\n')


if __name__ == '__main__':
    main()
