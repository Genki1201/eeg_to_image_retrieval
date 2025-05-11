import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_transformer_encoder(nn.Module):
    def __init__(self, in_channels=96, in_timestep=512, hidden_size=256, projection_dim=256, num_layers=1, nhead=4, dropout=0): # dropout増やしてみる
        super(EEG_transformer_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(in_timestep, hidden_size)  # 入力次元をhidden_sizeに変換
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=projection_dim, bias=False)

    def forward(self, x):
        x = self.input_projection(x)  # (batch_size, 96, 512) → (batch_size, 96, 256)
        x = x.transpose(0, 1)  # (batch_size, 96, 256) → (96, batch_size, 256)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # (batch_size, 256)
        x = self.fc(x)

        return x

class EEG_LSTM_original_model(nn.Module):
    def __init__(self, in_channels=96, n_features=512, projection_dim=256, num_layers=3):
        super(EEG_LSTM_original_model, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=256),
            nn.BatchNorm1d(num_features=256),          # BatchNormを挟む
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(in_features=256, out_features=256),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda() 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)

        # x = F.normalize(x, dim=-1)

        return x

class EEG_LSTM(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=256, num_layers=1):
        super(EEG_LSTM, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)
        # x = F.normalize(x, dim=-1)
        # print(x.shape, feat.shape)
        return x#, feat