import torch
import numpy as np
import torchvision
import torch.nn.functional


class ResBlock(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.upper_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(num_features=in_features),  # Normalization done on dataset
            torch.nn.Conv1d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features),
        )

        self.lower_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(num_features=in_features)  # Normalization done on dataset
        )

    def forward(self, x):
        # z: latent variable
        z = self.upper_layers.forward(x)
        z_prim = z + x  # (B, C, W, H)
        z_lower = self.lower_layers.forward(z_prim)
        return z_lower


class ResNet16(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResNet36(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResNet54(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResNet100(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResNet150(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240),
            ResBlock(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResBlockBN(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.upper_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=in_features),
            torch.nn.Conv1d(
                kernel_size=3, stride=1, padding=1,
                in_channels=in_features,
                out_channels=in_features),
        )

        self.lower_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=in_features)
        )

    def forward(self, x):
        # z: latent variable
        z = self.upper_layers.forward(x)
        z_prim = z + x  # (B, C, W, H)
        z_lower = self.lower_layers.forward(z_prim)
        return z_lower


class ResNet16BN(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)


class ResNet36BN(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, device, layernorm=False):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240),
            ResBlockBN(in_features=240)
        )

        self.fc = torch.nn.Linear(
            in_features=240*48,
            out_features=4
        )

    def forward(self, x, len):
        for l in self.layers:
            x = l.forward(x)

        x_2 = x.view((x.size(0), -1))  # (B, n_out*n_out*8)
        x_3 = self.fc.forward(x_2)

        return torch.nn.functional.softmax(x_3, dim=1)
