import torch
import math  # For P-LSTM
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence  # For Transformer
import numpy as np  # For Transformer
import torch.nn.functional as F  # For SotaLSTM


class ModelRNN(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff_1 = torch.nn.Linear(
            in_features=48,
            out_features=hidden_size
        )

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.ff_class = torch.nn.Linear(
            in_features=2*hidden_size,
            out_features=4
        )

    def forward(self, x, len):
        x_flat = self.ff_1.forward(x)
        cudnn_fmt = torch.nn.utils.rnn.PackedSequence(x_flat, len)
        hidden, cells = self.lstm.forward(cudnn_fmt[0])

        hidden = hidden.data

        h_mean = torch.mean(hidden, dim=1).squeeze()
        # h_max = torch.amax(hidden, dim=1).squeeze()
        h_max = torch.max(hidden, dim=1).values.squeeze()
        h_cat = torch.cat((h_max, h_mean), axis=1)
        logits = self.ff_class.forward(h_cat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


class ModelRNNLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff_1 = torch.nn.Linear(
            in_features=48,
            out_features=hidden_size
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.norm_2 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        self.ff_class = torch.nn.Linear(
            in_features=2*hidden_size,
            out_features=4
        )

    def forward(self, x, len):
        x_flat = self.ff_1.forward(x)
        x_norm = self.norm_1.forward(x_flat)
        cudnn_fmt = torch.nn.utils.rnn.PackedSequence(x_norm, len)
        hidden, cells = self.lstm.forward(cudnn_fmt[0])

        hidden = hidden.data

        hidden_norm = self.norm_2.forward(hidden)

        h_mean = torch.mean(hidden_norm, dim=1).squeeze()
        # h_max = torch.amax(hidden_norm, dim=1).squeeze()
        h_max = torch.max(hidden_norm, dim=1).values.squeeze()
        h_cat = torch.cat((h_max, h_mean), axis=1)
        logits = self.ff_class.forward(h_cat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


def fmod(a, b):
    return (b / math.pi) * torch.atan(torch.tan(math.pi * (a / b - 0.5))) + b / 2  # was: arctan


class PhasedLSTM(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            alpha=1e-3,
            tau_max=3.0,
            r_on=5e-2,
            device='cuda'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.alpha = alpha
        self.r_on = r_on
        self.device = device

        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.W = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv) # Test -1..1, default 0..1
        )

        self.U = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv) # Test -1..1, default 0..1
        )

        self.b = torch.nn.Parameter(
            torch.FloatTensor(4 * hidden_size).zero_()
        )

        self.w_peep = torch.nn.Parameter(
            torch.FloatTensor(3 * hidden_size).uniform_(-stdv, stdv)
        )

        self.tau = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, tau_max).exp_()
        )

        self.shift = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, torch.mean(self.tau).item())
        )

    def forward(self, x, h_c=None):
        if h_c is None:
            h = torch.zeros((x.size(0), self.hidden_size)).to(self.device)
            c = torch.zeros((x.size(0), self.hidden_size)).to(self.device)
        else:
            h, c = h_c
        h_out = []
        # cuda dnn
        # x => (B, Seq, F)
        x_seq = x.permute(1, 0, 2)  # (Seq, B, F)

        seq_len = x_seq.size(0)
        times = torch.arange(seq_len).unsqueeze(dim=1)  # (Seq, 1)
        times = times.expand((seq_len, self.hidden_size)).to(self.device)
        phi = fmod((times - self.shift), self.tau) / (self.tau + 1e-8)

        alpha = self.alpha
        if not self.training:  # model = model.eval() Dropout
            alpha = 0

        k = torch.where(
            phi < 0.5 * self.r_on,
            2.0 * phi / self.r_on,
            torch.where(
                torch.logical_and(0.5 * self.r_on <= phi, phi < self.r_on),
                2.0 - (2.0 * phi / self.r_on),
                alpha * phi
            )
        )

        for t, x_t in enumerate(x_seq):

            gates = torch.matmul(
                x_t.repeat(1, 3),
                self.W[:self.hidden_size*3, :self.hidden_size*3]  # (in, out)
            ) + torch.matmul(
                h.repeat(1, 3),
                self.U[:self.hidden_size*3, :self.hidden_size*3]
            ) + self.b[:self.hidden_size*3] + self.w_peep * c.repeat(1, 3)  # ? should this be c_t or/and c_{t-1}

            i_t = torch.sigmoid(gates[:, 0:self.hidden_size])
            f_t = torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2])
            o_t = torch.sigmoid(gates[:, self.hidden_size*2:self.hidden_size*3])

            gate_c = torch.matmul(
                x_t,
                self.W[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4]
            ) + torch.matmul(
                h,
                self.U[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4]
            ) + self.b[self.hidden_size*3:self.hidden_size*4]

            c_prim = f_t * c + i_t * torch.tanh(gate_c)
            c = k[t] * c_prim + (1 - k[t]) * c
            h_prim = torch.tanh(c_prim) * o_t
            h = k[t] * h_prim + (1 - k[t]) * h
            h_out.append(h)
        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)  # (B, Seq, F)
        return t_h_out


class ModelLSTM(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, alpha, tau_max, r_on):
        super().__init__()
        self.hiddenSize = hidden_size
        self.lstmLayers = lstm_layers
        self.device = device

        self.ff = torch.nn.Linear(
            in_features=48,
            out_features=self.hiddenSize
        )

        layers = []
        for _ in range(self.lstmLayers):
            layers.append(PhasedLSTM(
                input_size=self.hiddenSize,
                hidden_size=self.hiddenSize,
                device=self.device,
                alpha=alpha,
                tau_max=tau_max,
                r_on=r_on
            ))
        self.lstm = torch.nn.Sequential(*layers)

        self.fc = torch.nn.Linear(
            in_features=self.hiddenSize,
            out_features=4
        )

    def forward(self, x, len):
        x_flat = self.ff.forward(x)
        cudnn_fmt = torch.nn.utils.rnn.PackedSequence(x_flat, len)
        hidden = self.lstm.forward(cudnn_fmt[0])

        # hidden = hidden.data

        z_2 = torch.mean(hidden, dim=1).squeeze()  # B, Hidden_size
        logits = self.fc.forward(z_2)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim

        # # B, Seq, F => B, 28, 28
        # z_0 = self.ff.forward(x)
        # z_1 = self.lstm.forward(z_0)  # B, Seq, Hidden_size
        # z_2 = torch.mean(z_1, dim=1).squeeze()  # B, Hidden_size
        # logits = self.fc.forward(z_2)
        # y_prim = torch.softmax(logits, dim=1)
        # return y_prim


class ModelLSTMLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, alpha, tau_max, r_on):
        super().__init__()
        self.hiddenSize = hidden_size
        self.lstmLayers = lstm_layers
        self.device = device

        self.ff = torch.nn.Linear(
            in_features=48,
            out_features=self.hiddenSize
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        layers = []
        for _ in range(self.lstmLayers):
            layers.append(PhasedLSTM(
                input_size=self.hiddenSize,
                hidden_size=self.hiddenSize,
                device=self.device,
                alpha=alpha,
                tau_max=tau_max,
                r_on=r_on
            ))
        self.lstm = torch.nn.Sequential(*layers)

        self.norm_2 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        self.fc = torch.nn.Linear(
            in_features=self.hiddenSize,
            out_features=4
        )

    def forward(self, x, len):
        x_flat = self.ff.forward(x)
        x_norm = self.norm_1.forward(x_flat)
        cudnn_fmt = torch.nn.utils.rnn.PackedSequence(x_norm, len)
        hidden = self.lstm.forward(cudnn_fmt[0])

        hidden = hidden.data
        hidden_norm = self.norm_2.forward(hidden)

        z_2 = torch.mean(hidden_norm, dim=1).squeeze()  # B, Hidden_size
        logits = self.fc.forward(z_2)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim

        # # B, Seq, F => B, 28, 28
        # z_0 = self.ff.forward(x)
        # z_1 = self.lstm.forward(z_0)  # B, Seq, Hidden_size
        # z_2 = torch.mean(z_1, dim=1).squeeze()  # B, Hidden_size
        # logits = self.fc.forward(z_2)
        # y_prim = torch.softmax(logits, dim=1)
        # return y_prim


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, device, dropout, transformer_heads):
        super().__init__()
        self.hiddenSize = hidden_size
        self.dropout = dropout
        self.transformerHeads = transformer_heads
        self.device = device

        self.project_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.project_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.project_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=hidden_size)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x, lengths, atten):
        batch_size = x.size(0)  # x.shape (B, seq, HIDDEN_SIZE)
        seq_size = x.size(1)

        # shape (B, seq, hidden_size)
        k = self.project_k.forward(x)
        q = self.project_q.forward(x)
        v = self.project_v.forward(x)

        # shape (B, seq, Heads, HIDDEN_SIZE/Heads)
        # shape (B, Heads, seq, HIDDEN_SIZE/Heads)
        k = k.view(batch_size, seq_size, self.transformerHeads,
                   int(self.hiddenSize/self.transformerHeads)).transpose(1, 2)
        q = q.view(batch_size, seq_size, self.transformerHeads,
                   int(self.hiddenSize/self.transformerHeads)).transpose(1, 2)
        v = v.view(batch_size, seq_size, self.transformerHeads,
                   int(self.hiddenSize/self.transformerHeads)).transpose(1, 2)

        atten_raw = q @ k.transpose(-1, -2) / np.sqrt(x.size(-1))

        mask = torch.tril(torch.ones(seq_size, seq_size)).to(self.device)  # (Seq, Seq)
        atten_mask = atten_raw.masked_fill(mask == 0, value=float('-inf'))  # (B, Seq, Seq)
        for idx, length in enumerate(lengths):  # (B, Seq, Seq)
            atten_mask[idx, :, length:] = float('-inf')
            atten_mask[idx, length:, :] = float('-inf')

        atten = torch.softmax(atten_mask, dim=-1)
        atten = atten.masked_fill(((atten > 0) == False), value=0.0)
        out = atten @ v

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_size, self.hiddenSize)
        atten = atten.detach().mean(dim=1)  # shape (B, Heads, seq, seq) => (B, seq, seq)

        # torch.nn.Module > self.training
        # model.eval() model.train()
        out_1 = x + torch.dropout(out, p=self.dropout, train=self.training)
        out_1_norm = self.norm_1.forward(out_1)

        out_2 = self.ff.forward(out_1_norm)
        out_3 = out_1_norm + out_2
        y_prim = self.norm_2.forward(out_3)

        return y_prim, lengths, atten


class Transformer(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, dropout=0.1, transformer_heads=4):
        super().__init__()
        self.hiddenSize = hidden_size
        self.numLayers = lstm_layers
        self.device = device
        self.dropout = dropout
        self.transformer_heads = transformer_heads

        self.project_w_e = torch.nn.Embedding(
            num_embeddings=48,  # dataset_full.max_classes_tokens
            embedding_dim=self.hiddenSize
        )
        self.project_p_e = torch.nn.Embedding(
            num_embeddings=240,  # dataset_full.max_length
            embedding_dim=self.hiddenSize
        )

        self.transformer = torch.nn.ModuleList(
            [TransformerLayer(hidden_size=self.hiddenSize, device=device, dropout=self.dropout,
                              transformer_heads=self.transformer_heads) for _ in range(self.numLayers)]
        )

        self.fc = torch.nn.Linear(in_features=self.hiddenSize, out_features=4)

    def forward(self, x, len):
        x = pack_padded_sequence(x, len, batch_first=True)
        x_e = PackedSequence(
            data=self.project_w_e.forward(x.data.argmax(dim=1)),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True)

        # 0, 1, 2, 3, 4.... => (B, Seq) => project_p_e => (B, Seq, HIDDEN_SIZE)
        # lengths[0]
        pos_idxes = torch.arange(0, torch.max(lengths)).to(self.device)
        p_e = self.project_p_e.forward(pos_idxes)  # (Seq,)
        p_e = p_e.unsqueeze(dim=0)  # (1, Seq, H)
        p_e = p_e.expand(x_e_unpacked.size())

        z = x_e_unpacked + p_e
        atten = None
        for layer in self.transformer:
             z, lengths, atten = layer.forward(z, lengths, atten)

        z_2 = torch.mean(z, dim=1).squeeze()  # B, Hidden_size
        logits = self.fc.forward(z_2)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


class SotaLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.W = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv)  # Test -1..1, default 0..1
        )

        self.U = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv)  # Test -1..1, default 0..1
        )

        self.layer_norm_i = torch.nn.LayerNorm(
                hidden_size
        )
        self.layer_norm_f = torch.nn.LayerNorm(
                hidden_size
        )
        self.layer_norm_o = torch.nn.LayerNorm(
                hidden_size
        )
        self.layer_norm_c = torch.nn.LayerNorm(
                hidden_size
        )

        self.b = torch.nn.Parameter(
            torch.FloatTensor(4 * hidden_size).zero_()
        )
        # forget gate
        self.b.data[hidden_size:hidden_size*2].fill_(1.0)

        self.w_peep = torch.nn.Parameter(
            torch.FloatTensor(3 * hidden_size).uniform_(-stdv, stdv)
        )

        self.h_0 = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(-stdv, stdv)
        )
        self.c_0 = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(-stdv, stdv)
        )

    def forward(self, x, h_c=None):
        if h_c is None:
            h = self.h_0.expand(x.size(0), self.hidden_size)
            c = self.c_0.expand(x.size(0), self.hidden_size)
        else:
            h, c = h_c
        h_out = []
        # cuda dnn
        # x => (B, Seq, F)
        x_seq = x.permute(1, 0, 2)  # (Seq, B, F)
        for x_t in x_seq:

            gates = torch.matmul(
                x_t.repeat(1, 3),
                self.W[:self.hidden_size*3, :self.hidden_size*3] # (in, out)
            ) + torch.matmul(
                h.repeat(1, 3),
                self.U[:self.hidden_size*3, :self.hidden_size*3]
            ) + self.b[:self.hidden_size*3] + self.w_peep * c.repeat(1, 3)  # ? should this be c_t or/and c_{t-1}

            i_t = torch.sigmoid(self.layer_norm_i.forward(gates[:, 0:self.hidden_size]))
            f_t = torch.sigmoid(self.layer_norm_f.forward(gates[:, self.hidden_size:self.hidden_size*2]))
            o_t = torch.sigmoid(self.layer_norm_o.forward(gates[:, self.hidden_size*2:self.hidden_size*3]))

            gate_c = self.layer_norm_c.forward(torch.matmul(
                x_t,
                self.W[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4]
            ) + torch.matmul(
                h,
                self.U[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4]
            ) + self.b[self.hidden_size*3:self.hidden_size*4])

            d_h = torch.ones_like(h)
            d_c = torch.ones_like(c)
            if self.training:
                d_h = F.dropout(d_h, p=0.5)
                d_c = F.dropout(d_c, p=0.5)

            c_prim = f_t * c + i_t * torch.tanh(gate_c)
            c = d_c * c_prim + (1 - d_c) * c
            h_prim = torch.tanh(c) * o_t
            h = d_h * h_prim + (1 - d_h) * h

            h_out.append(h)
        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)  # (B, Seq, F)
        return t_h_out


class ModelSotaLSTM(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device):
        super().__init__()
        self.hiddenSize = hidden_size
        self.numLayers = lstm_layers
        self.device = device

        self.ff = torch.nn.Linear(
            in_features=48,
            out_features=self.hiddenSize
        )

        layers = []
        for _ in range(self.numLayers):
            layers.append(SotaLSTM(
                input_size=self.hiddenSize,
                hidden_size=self.hiddenSize,
            ))
        self.lstm = torch.nn.Sequential(*layers)

        self.fc = torch.nn.Linear(
            in_features=self.hiddenSize,
            out_features=4
        )

    def forward(self, x, len):
        # B, Seq, F => B, 28, 28
        z_0 = self.ff.forward(x)

        z_1_layers = []
        for i in range(self.numLayers):
            z_1_layers.append(self.lstm[i].forward(z_0))
            z_0 = z_1_layers[-1]  # + z_0

        # densnet => concat
        # resnet => sum
        # attenion
        z_1 = torch.stack(z_1_layers, dim=1)  # (B, Layers, Seq, Features)
        z_1 = torch.mean(z_1, dim=1)  # (B, Seq, Features)

        z_2 = torch.mean(z_1, dim=1).squeeze()  # (B, Features) Temporal pooling (Attention)
        logits = self.fc.forward(z_2)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


class ModelSotaLSTMLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device):
        super().__init__()
        self.hiddenSize = hidden_size
        self.numLayers = lstm_layers
        self.device = device

        self.ff = torch.nn.Linear(
            in_features=48,
            out_features=self.hiddenSize
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        layers = []
        for _ in range(self.numLayers):
            layers.append(SotaLSTM(
                input_size=self.hiddenSize,
                hidden_size=self.hiddenSize,
            ))
        self.lstm = torch.nn.Sequential(*layers)

        self.norm_2 = torch.nn.LayerNorm(normalized_shape=hidden_size)

        self.fc = torch.nn.Linear(
            in_features=self.hiddenSize,
            out_features=4
        )

    def forward(self, x, len):
        # B, Seq, F => B, 28, 28
        z_0 = self.ff.forward(x)

        z_0_n = self.norm_1.forward(z_0)

        z_1_layers = []
        for i in range(self.numLayers):
            z_1_layers.append(self.lstm[i].forward(z_0_n))
            z_0_n = z_1_layers[-1]  # + z_0

        z_1 = torch.stack(z_1_layers, dim=1)  # (B, Layers, Seq, Features)
        z_1_norm = self.norm_2.forward(z_1)
        z_1 = torch.mean(z_1_norm, dim=1)  # (B, Seq, Features)

        z_2 = torch.mean(z_1, dim=1).squeeze()  # (B, Features) Temporal pooling (Attention)
        logits = self.fc.forward(z_2)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim
