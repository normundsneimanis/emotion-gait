import torch


class ModelRNN(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff_1 = torch.nn.Linear(
            in_features=3*16,
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
        h_max = torch.amax(hidden, dim=1).squeeze()
        h_cat = torch.cat((h_max, h_mean), axis=1)
        logits = self.ff_class.forward(h_cat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


