import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiEncoder(nn.Module):
    '''Bidirectional Encoder

    Args:
        num_vocab: vocabulary size of input sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
    
    Attributes:
        hidden_size
        embed: nn.Embedding initialized by embedding_matrix
        bilstm: bidirectional LSTM
    '''

    def __init__(self,
                 num_vocab,
                 embedding_dim,
                 hidden_size,
                 embedding_matrix=None):
        super(BiEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_vocab, embedding_dim=embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            embedding_matrix = torch.from_numpy(embedding_matrix)
            self.embed.weight.data = embedding_matrix
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True)

    def forward(self, x, lengths):
        '''
        Args:
            x: input sequence (batch_size, seq_len)
            lengths: tensor that retains true lengths before padding. must be sorted. shape=(batch_size,)
        Returns:
            output: LSTM output, shape=(seq_len,batch_size,hidden_size)
            (h, c): LSTM states at last timestep, each shape is (1,batch_size,2*hidden_size)
        '''
        embed = self.embed(x)  #(batch_size, seq_len, embedding_dim)
        embed = pack_padded_sequence(
            embed, lengths, batch_first=True)  #PackedSequenceに変換
        assert embed[0].size(0) == torch.sum(lengths), '{},{}'.format(
            embed[0].size(0), torch.sum(lengths))

        output, (h, c) = self.bilstm(
            embed
        )  #(any_len, batch_size, 2*hidden_size), (2, batch_size, hidden_size)
        # reshape states into (1,batch_size, 2*hidden_size)
        h = h.permute(1, 2, 0).contiguous().view(1, -1, 2 * self.hidden_size)
        c = c.permute(1, 2, 0).contiguous().view(1, -1, 2 * self.hidden_size)
        #print(output[0].size())
        output, lengths = pad_packed_sequence(output)
        return output, (h, c)  #(seq_len, batch_size, hidden_size)
