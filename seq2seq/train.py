#!~/.pyenv/versions/anaconda3-5.0.0/bin/python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import pickle
import time
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from models import seq2seq
import utils

MODEL_PATH = './saved_models/'
FIGURE_PATH = './figures/'


def train(src,
          tgt,
          lengths,
          model,
          optimizer,
          criterion,
          tgt_vocab,
          is_train=True,
          teacher_forcing_ratio=0.8):
    """一回のミニバッチ学習
    Args:
        src (torch.Tensor): source data
        tgt (torch.Tensor): target data
        model: seq2seq model
        optimizer (torch.optim)
        criterion: loss function
        tgt_vocab (Vocab): target vocabulary
        is_train (bool): if True, parameters are upgraded by backpropagain. Default: True
        teacher_forcing_ratio (float): the probability of inputting ground truth(0.0~1.0)
    Returns:
        loss (float): averaged loss of all tokens
    """
    y_pred = model(src, tgt, lengths, tgt_vocab, teacher_forcing_ratio)

    loss = criterion(y_pred, tgt)
    _, y_pred = y_pred.max(1)  #(batch_size, seq_len)
    bleu = utils.calc_bleu(y_pred, tgt)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), bleu


def train_iters(model, criterion, train_dataloader, valid_dataloader,
                tgt_vocab, epochs, model_file):
    """Train Encoder-Decoder model
    Args:
        model: seq2seq model
        criterion: loss function
        train_dataloader (DataLoader): Dataloader of train data
        valid_dataloader (DataLoader): Dataloader of validation data
        tgt_vocab (Vocab): Vocab instance of target vocabulary
        epochs (int)
        model_file (string): the name of the file to save the model in
    Return:
        plot_losses (list of float): validation loss records
    TODO: save the model when bleu score is maximum
    """

    optimizer = optim.Adam(model.parameters())

    losses = []
    # bleuに関してbestなものを選んだ方がいいかも
    best_loss = np.inf
    best_bleu = 0.0
    num_epoch = valid_dataloader.size // valid_dataloader.batch_size

    for epoch in range(epochs):
        teacher_forcing_ratio = max(0, 1 - 1.5 * epoch / epochs)
        start = time.time()
        valid_loss = 0
        valid_bleu = 0
        model.train()
        for batch_id, (batch_X, batch_Y,
                       X_lengths) in enumerate(train_dataloader):
            loss, bleu = train(
                batch_X,
                batch_Y,
                X_lengths,
                model,
                optimizer,
                criterion,
                tgt_vocab,
                is_train=True,
                teacher_forcing_ratio=teacher_forcing_ratio)
            elapsed_sec = time.time() - start
            elapsed_min = elapsed_sec // 60
            elapsed_sec = elapsed_sec - 60 * elapsed_min
            print(
                '\rEpoch:{} Batch:{}/{} Loss:{:.4f} BLEU:{:.2f} Time:{:.0f}m{:.1f}s'
                .format(epoch, batch_id,
                        train_dataloader.size // train_dataloader.batch_size,
                        loss, bleu, elapsed_min, elapsed_sec),
                end='')
        print()
        model.eval()
        for batch_id, (batch_X, batch_Y,
                       X_lengths) in enumerate(valid_dataloader):
            loss, bleu = train(
                batch_X,
                batch_Y,
                X_lengths,
                model,
                optimizer,
                criterion,
                tgt_vocab,
                is_train=False,
                teacher_forcing_ratio=0)
            valid_loss += loss
            valid_bleu += bleu
            losses.append(loss)

        mean_valid_loss = valid_loss / num_epoch
        mean_valid_bleu = valid_bleu / num_epoch
        print('Valid Loss:{:.4f} Valid BLEU:{:.2f}'.format(
            mean_valid_loss, mean_valid_bleu))
        # save model when valid loss is minimum
        if mean_valid_bleu > best_bleu:
            best_bleu = mean_valid_bleu
            print('Saving model because valid BLEU score improved.')
            torch.save(model.state_dict(), MODEL_PATH + model_file)

    return losses


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--src_column', type=str)
    parser.add_argument('--tgt_column', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--src_maxlen', type=int)
    parser.add_argument('--tgt_maxlen', type=int)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--loss_file', type=str)
    parser.add_argument('--word2vec_path', type=str)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--attention')

    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size is not None else 32
    src_maxlen = args.src_maxlen if args.src_maxlen is not None else 100
    tgt_maxlen = args.tgt_maxlen if args.tgt_maxlen is not None else 100
    model_file = args.model_file if args.model_file is not None else 'tmp.model'
    loss_file = args.loss_file if args.loss_file is not None else 'tmp.png'
    data_path = args.data_path if args.data_path is not None else './data/train.csv'
    embedding_dim = args.embedding_dim if args.embedding_dim is not None else 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading data...')
    train = utils.DataReader()
    train.add_data_from_csv(data_path, 'src', 'tgt', preprocess=False)

    #長い系列のデータを削る
    train.drop_long_seq(src_maxlen, tgt_maxlen)

    src_lengths = train.data['src'].str.split().apply(len)
    src_lengths = np.array(src_lengths).astype('int32') - 1  #後で<EOS>を削除するため

    data_size = train.data_size

    # make dictionary
    src_words = utils.Vocab()
    tgt_words = utils.Vocab()

    for i in range(data_size):
        for word in (train.data.loc[i, 'src']).split():
            src_words.add(word)
        for word in (train.data.loc[i, 'tgt']).split():
            tgt_words.add(word)

    with open('src.vocab', 'wb') as f:
        pickle.dump(src_words, f)

    with open('tgt.vocab', 'wb') as f:
        pickle.dump(tgt_words, f)

    print('vocabulary size in source is', src_words.size)
    print('vocabulary size in target is', tgt_words.size)

    src_embedding_matrix = None
    tgt_embedding_matrix = None

    unknown_set = set()

    # Embedding層の初期値としてpre-trainさせたword2vec embeddingを用いる。
    # 単語辞書の中にはword2vecモデルに含まれない単語もあるので、そのembeddingは一様乱数で初期化する
    if args.word2vec_path is not None:

        print('Loading word2vec model...')

        word2vec = Word2Vec.load(args.word2vec_path)

        src_embedding_matrix = np.random.uniform(
            low=-0.05, high=0.05, size=(src_words.size, embedding_dim))
        tgt_embedding_matrix = np.random.uniform(
            low=-0.05, high=0.05, size=(tgt_words.size, embedding_dim))

        for i, word in enumerate(src_words):
            try:
                src_embedding_matrix[i] = word2vec[word]
            except KeyError:
                if word not in unknown_set:
                    unknown_set.add(word)
        for i, word in enumerate(tgt_words):
            try:
                tgt_embedding_matrix[i] = word2vec[word]
            except KeyError:
                if word not in unknown_set:
                    unknown_set.add(word)

        src_embedding_matrix[0] = np.zeros((embedding_dim, ))
        tgt_embedding_matrix[0] = np.zeros((embedding_dim, ))

        src_embedding_matrix = src_embedding_matrix.astype('float32')
        tgt_embedding_matrix = tgt_embedding_matrix.astype('float32')

        unknown_set.remove(utils.Vocab.pad_token)
        unknown_set.remove(utils.Vocab.bos_token)
        unknown_set.remove(utils.Vocab.eos_token)
        unknown_set.remove(utils.Vocab.unk_token)
        unknown_set.remove(utils.Vocab.num_token)
        unknown_set.remove(utils.Vocab.alp_token)

    src = np.zeros((data_size, src_maxlen), dtype='int32')
    tgt = np.zeros((data_size, tgt_maxlen), dtype='int32')

    for i in range(data_size):
        for j, word in enumerate(train.data.loc[i, 'src'].split()):
            if word in unknown_set:
                word = utils.Vocab.unk_token
            src[i][j] = src_words.word2id(word)
        for j, word in enumerate(train.data.loc[i, 'tgt'].split()):
            if word in unknown_set:
                word = utils.Vocab.unk_token
            tgt[i][j] = tgt_words.word2id(word)

    src = src[:, :-1]  # <EOS>削除
    tgt = tgt[:, 1:]  # <BOS>削除

    with codecs.open('unknown.json', 'w', 'utf-8') as f:
        unknown_list = list(unknown_set)
        dump = json.dumps(unknown_list, ensure_ascii=False)
        f.write(dump)

    # not to include <PAD> in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=utils.Vocab.pad_id)

    params = {
        'src_num_vocab': src_words.size,
        'tgt_num_vocab': tgt_words.size,
        'embedding_dim': embedding_dim,
        'hidden_size': 512,
        'src_embedding_matrix': src_embedding_matrix,
        'tgt_embedding_matrix': tgt_embedding_matrix
    }

    with open('params.json', 'w') as f:
        json.dump(params, f)

    if args.attention is None:
        model = seq2seq.EncoderDecoder(**params).to(device)
    else:
        model = seq2seq.GlobalAttentionEncoderDecoder(**params).to(device)

    train_src, valid_src, train_tgt, valid_tgt, train_src_lengths, valid_src_lengths = train_test_split(
        src, tgt, src_lengths, test_size=0.1)
    train_dataloader = utils.DataLoader(
        train_src, train_tgt, train_src_lengths, batch_size=batch_size)
    valid_dataloader = utils.DataLoader(
        valid_src, valid_tgt, valid_src_lengths, batch_size=batch_size)

    print('Start Training')
    losses = train_iters(
        model,
        criterion,
        train_dataloader,
        valid_dataloader,
        src_words,
        epochs=30,
        model_file=model_file)

    plt.figure(figsize=(20, 8))
    plt.ylim(0, max(losses) + 1)
    plt.plot(losses)
    plt.savefig(FIGURE_PATH + loss_file)

    print('Finished!')


if __name__ == '__main__':
    main()
