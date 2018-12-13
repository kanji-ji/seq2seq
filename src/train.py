# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import math
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
          is_train=True,
          teacher_forcing_ratio=0.8):
    """one step minibatch training
    Args:
        src (tensor): source data
        tgt (tensor): target data
        model (child class of nn.Module): seq2seq model
        optimizer (torch.optim)
        criterion: loss function
        is_train (bool): if True, parameters are upgraded by backpropagain. Default: True
        teacher_forcing_ratio (float): the probability of inputting ground truth Default:[0, 1]
    Returns:
        loss (float): averaged loss of all tokens
    """
    y_pred = model(src, tgt, lengths, teacher_forcing_ratio)

    loss = criterion(y_pred, tgt)
    bleu = 0

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        tgt = tgt.tolist()
        _, y_pred = y_pred.max(1)  # shape = (batch_size, seq_len)
        y_pred = y_pred.tolist()
        bleu = utils.calc_bleu(y_pred, tgt)

    return loss.item(), bleu


def train_iters(model, criterion, train_dataloader, valid_dataloader, epochs,
                model_file):
    """Train Encoder-Decoder model
    Args:
        model (child class of nn.Module): seq2seq model
        criterion: loss function
        train_dataloader (DataLoader): Dataloader of train data
        valid_dataloader (DataLoader): Dataloader of validation data
        epochs (int)
        model_file (string): the name of the file to save the model in
    Return:
        losses (list of float): validation loss records
    TODO: save the model when BLEU score is maximum
    """

    optimizer = optim.Adam(model.parameters())

    losses = []
    best_bleu = 0.0

    for epoch in range(epochs):
        teacher_forcing_ratio = max(0, 1 - 1.5 * epoch / epochs)
        start = time.time()
        valid_loss = 0
        valid_bleu = 0
        num_batch = math.ceil(
            train_dataloader.size / train_dataloader.batch_size)

        model.train()
        for batch_id, (batch_X, batch_Y,
                       X_lengths) in enumerate(train_dataloader):
            loss, _ = train(
                batch_X,
                batch_Y,
                X_lengths,
                model,
                optimizer,
                criterion,
                is_train=True,
                teacher_forcing_ratio=teacher_forcing_ratio)
            elapsed_sec = time.time() - start
            elapsed_min = elapsed_sec // 60
            elapsed_sec = elapsed_sec - 60 * elapsed_min
            print(
                '\rEpoch:{} Batch:{}/{} Loss:{:.4f} Time:{:.0f}m{:.1f}s'.
                format(epoch + 1, batch_id + 1, num_batch, loss, elapsed_min,
                       elapsed_sec),
                end='')
        print()

        num_batch = math.ceil(
            valid_dataloader.size / valid_dataloader.batch_size)
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
                is_train=False,
                teacher_forcing_ratio=0)
            valid_loss += loss
            valid_bleu += bleu
            losses.append(loss)

        mean_valid_loss = valid_loss / num_batch
        mean_valid_bleu = valid_bleu / num_batch
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
    parser.add_argument('--attention', action='store_true')

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
    train = utils.DataBuilder()
    train.add_data_from_csv(data_path, 'src', 'tgt', preprocess=False)

    train.drop_long_seq(src_maxlen, tgt_maxlen)

    # make dictionary
    src_vocab, tgt_vocab = train.build_vocab()

    with open('./cache/src.vocab', 'wb') as f:
        pickle.dump(src_vocab, f)

    with open('./cache/tgt.vocab', 'wb') as f:
        pickle.dump(tgt_vocab, f)

    print('vocabulary size in source is', src_vocab.size)
    print('vocabulary size in target is', tgt_vocab.size)

    src_embedding_matrix = None
    tgt_embedding_matrix = None

    unknown_set = set()

    # use pre-trained word2vec embedding as Embedding Layer
    # if a word is not in word2vec model, its embedding initializes from uniform random number.
    if args.word2vec_path is not None:

        print('Loading word2vec model...')

        word2vec = Word2Vec.load(args.word2vec_path)

        assert embedding_dim == word2vec.size, 'embedding dim unmatched. args:{}, word2vec:{}'.format(
            embedding_dim, word2vec.size)

        src_embedding_matrix, src_unknown_set = utils.get_embedding_matrix(
            src_vocab, word2vec)
        tgt_embedding_matrix, tgt_unknown_set = utils.get_embedding_matrix(
            tgt_vocab, word2vec)

        unknown_set = src_unknown_set | tgt_unknown_set

    def replace_unknown(text):
        text = utils.replace_unknown(text, unknown_set)
        return text

    train.data = train.data.applymap(replace_unknown)

    src, tgt = train.make_id_array(src_maxlen, tgt_maxlen)
    src_lengths = train.data['src'].str.split().apply(len)
    src_lengths = np.array(src_lengths).astype('int32') - 1  #'cause delete <EOS> later.
    src = src[:, :-1]  # <EOS> delete
    tgt = tgt[:, 1:]  # <BOS> delete

    #dump unknown words as json file
    with codecs.open('./cache/unknown.json', 'w', 'utf-8') as f:
        unknown_list = list(unknown_set)
        dump = json.dumps(unknown_list, ensure_ascii=False)
        f.write(dump)

    # not to include <PAD> in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=utils.Vocab.pad_id)

    params = {
        'src_num_vocab': src_vocab.size,
        'tgt_num_vocab': tgt_vocab.size,
        'embedding_dim': embedding_dim,
        'hidden_size': 512,
        'src_embedding_matrix': src_embedding_matrix,
        'tgt_embedding_matrix': tgt_embedding_matrix
    }

    with open('./cache/params.json', 'w') as f:
        json.dump(params, f)

    if args.attention:
        model = seq2seq.GlobalAttentionEncoderDecoder(**params).to(device)
    else:
        model = seq2seq.EncoderDecoder(**params).to(device)

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
        epochs=30,
        model_file=model_file)

    plt.figure(figsize=(16, 6))
    plt.ylim(0, max(losses) + 1)
    plt.plot(losses)
    plt.savefig(FIGURE_PATH + loss_file)

    print('Finished!')


if __name__ == '__main__':
    main()
