import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #should be delete in the future

class Decoder(nn.Module):
    '''NN decoding from encoder's last states
    Args:
        num_vocab: vocabulary size of target sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
    Attributes:
        embed: nn.Embedding initialized by embedding_matrix
        lstm: nn.LSTM
        linear: nn.Linear (*, hidden_size)->(*, num_vocab)
    '''

    def __init__(self, num_vocab, embedding_dim, hidden_size,
                 embedding_matrix):
        super(Decoder, self).__init__()
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, num_vocab)

    def forward(self, decoder_input, decoder_states):
        '''
        Args:
            decoder_input: tensor (batch_size, seq_len)
            decoder_states(tuple): LSTM's initial state,each shape = (1, batch_size, hidden_dim)
            
        Returns:
            output: Decoder output shape=(seq_len,batch_size,num_vocab)
            hidden: tuple of last states, both shape=(1,batch_size,hidden_dim)
        '''
        embed = self.embed(decoder_input)  #(batch_size,seq_len,embedding_dim)
        assert len(embed.size()) == 3, '{}'.format(embed.size())
        output, hidden = self.lstm(
            embed.permute(1, 0, 2), decoder_states
        )  #(seq_len,batch_size,hidden_dim),(1,batch_size,hidden_dim)
        output = self.linear(output)  #(seq_len,batch_size,num_vocab)

        return output, hidden  # (seq_len,batch_size,num_vocab), tuple of (1,batch_size,hidden_dim)

    
class GlobalAttentionDecoder(nn.Module):
    '''Decoder using Global Attention mechanism
     Args:
        num_vocab: vocabulary size of target sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
        dropout_p: probability of dropout occurrence, Default:0.2(not implemented yet)
    Attributes:
        embed: nn.Embedding initialized by embedding_matrix
        lstm: nn.LSTM
        out: nn.Linear (*, )
    '''

    def __init__(self,
                 num_vocab,
                 embedding_dim,
                 hidden_size,
                 embedding_matrix,
                 dropout_p=0.2):
        super(GlobalAttentionDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.embed = nn.Embedding(
            num_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        #self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=self.hidden_size)
        self.out = nn.Linear(2 * hidden_size, num_vocab)

    def forward(self, decoder_input, hidden, encoder_outputs,mask=None):
        '''
        Args:
            decoder_input: (batch_size, seq_len),seq_len must be 1
            hidden(tuple): LSTM initial state, shape=(1,batch_size,hidden_size)
            encoder_outputs: (seq_len,batch_size,hidden_size)
            mask: each element must be 1 or 0. if 0, corresponding source word won't be paid attention to.(batch_size, seq_len)
        Returns:
            output: Decoder output, shape=(1,batch_size,num_vocab)
            hidden(tuple): LSTM last states, each shape=(1,batch_size,hidden_size)
            attn_weights: attention score of each timesteps, shape=(batch_size,seq_len,1)
        '''
        seq_len = decoder_input.size(1)
        assert seq_len == 1
        embed = self.embed(
            decoder_input
        )  # (batch_size, seq_len, embedding_dim), and seq_len must be 1
        embed = embed.permute(1, 0, 2)  # (seq_len,batch_size,embedding_dim)
        #embed = self.dropout(embed)

        output, (h, c) = self.lstm(
            embed, hidden
        )  #(seq_len,batch_size,hidden_size),tuple of (1,batch_size,hidden_size)
        
        attn_scores = encoder_outputs.permute(1, 0, 2).bmm(h.permute(
            1, 2, 0))  #内積注意，(batch_size,seq_len,1)
        
        if mask is not None:
            attn_scores.data.masked_fill_(mask.unsqueeze(2), -float('inf'))
        attn_weights = torch.softmax(
            attn_scores, dim=1)  #(batch_size,seq_len,1)
        context = encoder_outputs.permute(1, 2, 0).bmm(
            attn_weights)  #(batch_size,hidden_size,1)
        output = torch.cat(
            [context.permute(2, 0, 1), h],
            dim=2)  #(1,batch_size,2*hidden_size)
        output = self.out(output)  #(1, batch_size, num_vocab)
        return output, (h, c), attn_weights


class BahdanauAttentionDecoder(nn.Module):
    '''Decoder using Bahdanau attention
    Args:
        num_vocab: vocabulary size of target sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
        dropout_p: probability of dropout occurrence, Default:0.2
    '''

    def __init__(self,
                 num_vocab,
                 embedding_dim,
                 hidden_size,
                 embedding_matrix,
                 dropout_p=0.2):
        super(BahdanauAttentionDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.embed = nn.Embedding(
            num_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        #self.dropout = nn.Dropout(self.dropout_p)
        self.hidden_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        dot_vector = torch.randn((self.hidden_size,), dtype=torch.float32, device=device) / self.hidden_size
        self.dot_vector = nn.Parameter(dot_vector)
        self.lstm = nn.LSTM(
            input_size=embedding_dim+self.hidden_size, hidden_size=self.hidden_size)
        self.out = nn.Linear(hidden_size, num_vocab)

    def forward(self, decoder_input, hidden, encoder_outputs, mask=None):
        '''
        Args:
            decoder_input: (batch_size, seq_len),seq_len must be 1
            hidden(tuple): LSTM initial state, each shape = (1,batch_size,hidden_size)
            encoder_outputs: (seq_len,batch_size,hidden_size)
            mask: each element must be 1 or 0. if 0, corresponding source word won't be paid attention to.(batch_size, seq_len)
        Returns:
            output: Decoder output, shape=(1,batch_size,num_vocab)
            hidden(tuple): LSTM last states, each shape = (1,batch_size,hidden_size)
            attn_weights: attention score of each timesteps, shape = (batch_size,seq_len,1)
        '''
        h, c = hidden
        seq_len = encoder_outputs.size(0)
        embed = self.embed(decoder_input)  #(batch_size,1,embedding_dim)
        attn_hidden = self.attn_linear(encoder_outputs).permute(
            1, 2, 0)  #(batch_size,hidden_size,seq_len)
        hidden_hidden = self.hidden_linear(h).squeeze(0) #(batch_size,hidden_size)
        hidden_hidden = hidden_hidden.repeat(seq_len, 1, 1).permute(1, 2, 0)
        attn_hidden += hidden_hidden
        attn_hidden = torch.tanh(attn_hidden)
        attn_scores = torch.matmul(
            attn_hidden.permute(0, 2, 1),
            self.dot_vector)  #(batch_size,seq_len)
        if mask is not None:
            attn_scores.data.masked_fill_(mask, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(encoder_outputs.permute(
            1, 2, 0), attn_weights.unsqueeze(2)).permute(2, 0, 1)  #(1,batch_size,hidden_size)
        output, hidden = self.lstm(
            torch.cat((embed.permute(1, 0, 2), context), dim=2), hidden)
        output = self.out(output)  #(1,batch_size,num_vocab)
        return output, hidden, attn_weights