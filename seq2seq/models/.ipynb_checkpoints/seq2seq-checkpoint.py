import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .encoders import *
from .decoders import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #should be deleted in the future

class EncoderDecoder(nn.Module):
    '''Encoder-Decoder as a single module
    Args:
        src_num_vocab:vocabulary size of source sentences
        tgt_num_vocab:vocabulary size of target sentences
        embedding_dim:word embedding dimension
        hidden_size:hidden cell dimension of LSTM
        src_embeddding_matrix:initial values of source word embedding matrix
        tgt_embedding_matrix:initial values of target word embedding matrix
    '''

    def __init__(self, src_num_vocab, tgt_num_vocab, embedding_dim,
                 hidden_size, src_embedding_matrix, tgt_embedding_matrix):
        super(EncoderDecoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = Decoder(tgt_num_vocab, embedding_dim, 2 * hidden_size,
                               tgt_embedding_matrix)

    def forward(self, src, tgt, lengths, dec_vocab, teacher_forcing_ratio=0.8):
        output, encoder_states = self.encoder(src, lengths)

        tgt_length = tgt.size(1)  #tgt.shape(batch_size, seq_len)
        batch_size = tgt.size(0)

        outputs = []

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size,1)

        for i in range(tgt_length):
            is_teacher_forcing = True if np.random.random(
            ) < teacher_forcing_ratio else False
            output, decoder_states = self.decoder(
                decoder_input, decoder_states)  # (1,batch_size,vocab_size)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(output)
            if is_teacher_forcing:
                decoder_input = tgt[:, i].unsqueeze(1)
            else:
                #topi.detach()
                decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  #(batch_size,vocab_size,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length, dec_vocab):
        output, encoder_states = self.encoder(src, lengths)
        batch_size = src.size(0)
        outputs = []

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size,1)

        for i in range(tgt_length):
            output, decoder_states = self.decoder(
                decoder_input, decoder_states)  # (1,batch_size,vocab_size)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(topi)  # greedy search
            decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 0)  #(batch_size,seq_len)
        return outputs


class GlobalAttentionEncoderDecoder(nn.Module):
    '''Encoder+GlobalAttentionDecoder as a single class
    '''

    def __init__(self,
                 src_num_vocab,
                 tgt_num_vocab,
                 embedding_dim,
                 hidden_size,
                 src_embedding_matrix,
                 tgt_embedding_matrix,
                 dropout_p=0.2,
                 use_mask=True):
        super(GlobalAttentionEncoderDecoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = GlobalAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        self.use_mask = use_mask

    def forward(self, src, tgt, lengths, dec_vocab, teacher_forcing_ratio):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        tgt_length = tgt.size(1)
        batch_size = tgt.size(0)
        
        mask = None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            mask = torch.eq(src,0) #(batch_size, seq_len), paddingされた部分は参照しないようにしている

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size, 1)

        outputs = []

        for i in range(tgt_length):
            is_teacher_forcing = True if np.random.random(
            ) < teacher_forcing_ratio else False
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(output)
            if is_teacher_forcing:
                decoder_input = tgt[:, i].unsqueeze(1)
            else:
                #topi.detach()
                decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  #(batch_size,vocab_size,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length, dec_vocab):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        
        src = pack_padded_sequence(src, lengths, batch_first=True)
        src, _ = pad_packed_sequence(src, batch_first=True)
        mask = torch.ne(src,0) #(batch_size, seq_len)

        batch_size = src.size(0)
        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size, 1)

        outputs = []

        for i in range(tgt_length):
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(topi)  #greedy search
            decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 0)  #(batch_size,seq_len)
        return outputs


class BahdanauAttentionEncoderDecoder(nn.Module):
    '''BiEncoder and Bahadanau attention Decoder as a single class
    '''

    def __init__(self,
                 src_num_vocab,
                 tgt_num_vocab,
                 embedding_dim,
                 hidden_size,
                 src_embedding_matrix,
                 tgt_embedding_matrix,
                 dropout_p=0.2,
                 use_mask=True):
        super(BahdanauAttentionEncoderDecoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = BahdanauAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        self.use_mask = use_mask

    def forward(self, src, tgt, lengths, dec_vocab, teacher_forcing_ratio):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        tgt_length = tgt.size(1)
        batch_size = tgt.size(0)
        
        mask = None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            mask = torch.eq(src,0) #(batch_size, seq_len)

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size, 1)

        outputs = []

        for i in range(tgt_length):
            is_teacher_forcing = True if np.random.random(
            ) < teacher_forcing_ratio else False
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(output)
            if is_teacher_forcing:
                decoder_input = tgt[:, i].unsqueeze(1)
            else:
                #topi.detach()
                decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  #(batch_size,vocab_size,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length, dec_vocab):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        
        src = pack_padded_sequence(src, lengths, batch_first=True)
        src, _ = pad_packed_sequence(src, batch_first=True)
        mask = torch.ne(src,0) #(batch_size, seq_len)

        batch_size = src.size(0)
        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  #(batch_size, 1)

        outputs = []

        for i in range(tgt_length):
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(topi)  #greedy search
            decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 0)  #(batch_size,seq_len)
        return outputs