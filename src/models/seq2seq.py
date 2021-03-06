import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .encoders import *
from .decoders import *

sys.path.append(os.pardir)
import utils


class EncoderDecoder(nn.Module):
    """Encoder-Decoder as a single module
    Args:
        src_num_vocab:vocabulary size of source sentences
        tgt_num_vocab:vocabulary size of target sentences
        embedding_dim:word embedding dimension
        hidden_size:hidden cell dimension of LSTM
        src_embeddding_matrix:initial values of source word embedding matrix
        tgt_embedding_matrix:initial values of target word embedding matrix
    """

    def __init__(self, src_num_vocab, tgt_num_vocab, embedding_dim,
                 hidden_size, src_embedding_matrix, tgt_embedding_matrix):
        super(EncoderDecoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = Decoder(tgt_num_vocab, embedding_dim, 2 * hidden_size,
                               tgt_embedding_matrix)

    def forward(self, src, tgt, lengths, teacher_forcing_ratio=0.8):
        output, encoder_states = self.encoder(src, lengths)

        tgt_length = tgt.size(1)  # tgt.size() = (batch_size, seq_len)
        batch_size = src.size(0)

        outputs = []

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [utils.Vocab.bos_id] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  # (batch_size,1)

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
                # topi.detach()
                decoder_input = topi.permute(1, 0)  # (batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  # (batch_size,num_vocab,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length):
        tgt = torch.empty(1, tgt_length)  # dummy variable
        outputs = self.forward(src, tgt, lengths, teacher_forcing_ratio=0)
        _, outputs = outputs.max(1)
        return outputs


class GlobalAttentionEncoderDecoder(nn.Module):
    """Encoder+GlobalAttentionDecoder as a single class
    """

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
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = GlobalAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        self.use_mask = use_mask

    def forward(self, src, tgt, lengths, teacher_forcing_ratio):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        tgt_length = tgt.size(1)
        batch_size = src.size(0)

        mask = None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            # (batch_size, seq_len), paddingされた部分は参照しないようにしている
            mask = torch.eq(src, 0)

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [utils.Vocab.bos_id] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  # (batch_size, 1)

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
                # topi.detach()
                decoder_input = topi.permute(1, 0)  # (batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  # (batch_size,vocab_size,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length):
        tgt = torch.empty(1, tgt_length)
        outputs = self.forward(src, tgt, lengths, teacher_forcing_ratio=0)
        _, outputs = outputs.max(1)
        return outputs


class GlobalAttentionBeamEncoderDecoder(nn.Module):
    """Encoder+GlobalAttentionDecoder as a single class(beam search version)
    Attributes:
        beam_size (int): Default: 4 
    """

    def __init__(self,
                 src_num_vocab,
                 tgt_num_vocab,
                 embedding_dim,
                 hidden_size,
                 src_embedding_matrix,
                 tgt_embedding_matrix,
                 dropout_p=0.2,
                 beam_size=4,
                 use_mask=True):
        super(GlobalAttentionBeamEncoderDecoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = GlobalAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        self.beam_size = beam_size
        self.use_mask = use_mask

    def forward(self, src, tgt, lengths, teacher_forcing_ratio):
        """
        Args:
            src:
            tgt:
            lengths:
            dec_vocab:
            teacher_forcing_ratio
        Returns:
            top_k_outputs (tensor): shape=(batch_size, beam_size, vocab_size, seq_len)
            top_k_seq (tensor): shape=(batch_size, beam_size, seq_len)
        """

        encoder_outputs, encoder_states = self.encoder(src, lengths)

        # get scalar values
        tgt_length = tgt.size(1)
        batch_size = src.size(0)
        hidden_size = encoder_states[0].size(-1)

        mask = None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            # (batch_size, seq_len), paddingされた部分は参照しないようにしている
            mask = torch.eq(src, 0)

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [utils.Vocab.bos_id] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  # (batch_size, 1)
        '''
        list element is tuple of (output, index)
        index is tensor whose values are from 0 to k-1.
        This tensor indexes output tensor one timestep before.
        Tracing back this list along index tensor form final k output tensors.
        '''
        outputs = []

        output, decoder_states, attn_weights = self.decoder(
            decoder_input, decoder_states, encoder_outputs, mask)
        _, top_k_words = torch.topk(output, self.beam_size,
                                    2)  # (1, batch_size, k)

        # repeat decoder_states k times
        h, c = decoder_states  # (1, batch_size, hidden_size)
        h = h.expand(self.beam_size, batch_size,
                     hidden_size).contiguous().view(1, -1, hidden_size)
        c = c.expand(self.beam_size, batch_size,
                     hidden_size).contiguous().view(1, -1, hidden_size)
        decoder_states = (h, c)  # (1, k*batch_size, hidden_size)

        # append output tensor.repeat output tensor k times to adjust
        # output tensor to be appended after first timestep.
        # index tensor is empty because this is first timestep
        # (k, batch_size, num_vocab), (k, batch_size)
        outputs.append((output.expand(self.beam_size,
                                      batch_size, output.size(2)),
                        top_k_words.squeeze(0).permute(1, 0)))

        log_probs = F.log_softmax(
            output, dim=-1).gather(2,
                                   top_k_words).squeeze(0)  # (batch_size, k)

        # (batch_size, seq_len) -> (k * batch_size, seq_len)
        mask = mask.unsqueeze(0).expand(self.beam_size, - 1, -1)
        mask = mask.contiguous().view(self.beam_size * batch_size, -1)
        # (seq_len, batch_size, hidden_size) -> (seq_len, k * batch_size, hidden_size)
        encoder_outputs = encoder_outputs.unsqueeze(1).expand(
            -1, self.beam_size, -1, -1).contiguous().view(
                -1, self.beam_size * batch_size, hidden_size)

        for i in range(tgt_length - 1):
            # Strictly, random value must be generated indepedently
            # for each data, each timestep, each beam_index
            is_teacher_forcing = True if np.random.random(
            ) < teacher_forcing_ratio else False
            if is_teacher_forcing:
                decoder_input = tgt[:, i].repeat(
                    self.beam_size).contiguous().view(-1,
                                                      1)  # (k*batch_size, 1)
            else:
                # top_k_words.detach() necessary?
                # be careful batch_size*k and k*batch_size are different
                assert top_k_words.size(
                    2
                ) == self.beam_size, 'top_k_words shape is differently permuted'
                decoder_input = top_k_words.permute(0, 2, 1).contiguous().view(
                    1, -1).permute(1, 0)  # (k*batch_size, 1)

            # output:(1, k*batch_size, num_vocab)
            # decoder_states:(1, k*batch_size, hidden_size)
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            _, top_k_words = torch.topk(output, self.beam_size,
                                        2)  # (1, k*batch_size, k)

            # compute conditional probs to take top-k words by log_probs.
            # (1, k*batch_size, k)
            c_probs = F.log_softmax(output, dim=-1).gather(2, top_k_words)

            # permute tensor shape to take top-k values
            # confirm first k is that of one timestep before, second k is present one.
            # shape = (batch_size, k, k)
            # first k is that of one timestep before, second is present one.
            log_probs = log_probs.unsqueeze(2).expand(
                -1, -1, self.beam_size) + c_probs.squeeze(0).contiguous().view(
                    self.beam_size, -1, self.beam_size).permute(1, 0, 2)

            # (batch_size, k), index can be from 0 to k*k-1
            log_probs, top_k_index = torch.topk(
                log_probs.contiguous().view(
                    -1, self.beam_size * self.beam_size), self.beam_size, -1)

            # (1, k, batch_size, k) decompose k*batch_size to (k, batch_size)
            top_k_words = top_k_words.contiguous().view(
                1, self.beam_size, -1, self.beam_size)
            top_k_words = top_k_words.permute(0, 2, 1, 3).contiguous().view(
                1, -1, self.beam_size * self.beam_size)  # (1, batch_size, k*k)
            # take top-k words, this is decoder_input at next timestep
            top_k_words = top_k_words.squeeze(0).gather(
                1, top_k_index).unsqueeze(0)  # (1, batch_size, k)

            top_k_index = top_k_index.permute(1, 0)
            # next, take output tensor
            # (k, batch_size, num_vocab)
            output = output.contiguous().view(self.beam_size, batch_size, -1)
            output = output.gather(
                0,
                top_k_index.unsqueeze(2).expand(self.beam_size, batch_size,
                                                output.size(2)) // 4)

            outputs.append((output, top_k_words.squeeze(0).permute(1, 0),
                            top_k_index // 4))

            # next, take states tensor to feed at next timestep
            decoder_states = list(decoder_states)

            for i, state in enumerate(decoder_states):
                # (k, batch_size, hidden_size)
                state = state.contiguous().view(self.beam_size, batch_size, -1)
                state = state.gather(
                    0,
                    top_k_index.unsqueeze(2).expand(self.beam_size, batch_size,
                                                    state.size(2)) // 4)
                state = state.unsqueeze(0)  # (1, k, batch_size, hidden_size)
                state = state.contiguous().view(1, -1, hidden_size)
                decoder_states[i] = state

            decoder_states = tuple(decoder_states)

        # tracing back phase
        # (1, k, batch_size, num_vocab)
        top_k_outputs = []
        # (1, k, batch_size)
        top_k_seq = []

        output, top_k_words, prev_index = outputs[-1]
        top_k_outputs.append(output.unsqueeze(0))
        top_k_seq.append(top_k_words.unsqueeze(0))

        for i in range(2, tgt_length + 1):
            if i != tgt_length:
                output, top_k_words, next_index = outputs[-i]
                output = output.gather(
                    0,
                    prev_index.unsqueeze(2).expand(self.beam_size, batch_size,
                                                   output.size(2)))
                next_index = next_index.gather(0, prev_index)
                top_k_words = top_k_words.gather(0, prev_index)
                top_k_outputs.insert(0, output.unsqueeze(0))
                top_k_seq.insert(0, top_k_words.unsqueeze(0))
                prev_index = next_index
            else:
                output, top_k_words = outputs[-i]
                output = output.gather(
                    0,
                    prev_index.unsqueeze(2).expand(self.beam_size, batch_size,
                                                   output.size(2)))
                top_k_words = top_k_words.gather(0, prev_index)
                top_k_outputs.insert(0, output.unsqueeze(0))
                top_k_seq.insert(0, top_k_words.unsqueeze(0))

        top_k_outputs = torch.cat(
            top_k_outputs,
            dim=0).permute(2, 1, 3, 0)  # (batch_size, k, vocab_size, seq_len)
        top_k_seq = torch.cat(
            top_k_seq, dim=0).permute(2, 1, 0)  # (batch_size, k, seq_len)

        return top_k_outputs[:, 0], top_k_seq[:, 0]

    def sample(self, src, lengths, tgt_length):
        tgt = torch.empty(1, tgt_length)
        _, top_k_seq = self.forward(src, tgt, lengths, teacher_forcing_ratio=0)
        return top_k_seq  # (batch_size, k, seq_len)


class BahdanauAttentionEncoderDecoder(nn.Module):
    """BiEncoder and Bahadanau attention Decoder as a single class
    """

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
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = BahdanauAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        self.use_mask = use_mask

    def forward(self, src, tgt, lengths, teacher_forcing_ratio):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        tgt_length = tgt.size(1)
        batch_size = src.size(0)

        mask = None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            mask = torch.eq(src, 0)  # (batch_size, seq_len)

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [utils.Vocab.bos_id] * batch_size,
            dtype=torch.long,
            device=self.device).unsqueeze(1)  # (batch_size, 1)

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
                # topi.detach()
                decoder_input = topi.permute(1, 0)  # (batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  # (batch_size,vocab_size,seq_len)
        return outputs

    def sample(self, src, lengths, tgt_length):
        tgt = torch.empty(1, tgt_length)
        outputs = self.forward(src, tgt, lengths, teacher_forcing_ratio=0)
        _, outputs = outputs.max(1)
        return outputs
