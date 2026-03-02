import torch
from torch import nn
import math

class xrd_encoder_T(nn.Module):
    def __init__(self, hidden_size=128, seq_len=750, num_layers=4, num_heads=4, dropout=0.1, latent_size=256):
        super().__init__()
        self.position_encoding = self.generate_positional_encoding(seq_len, hidden_size)
        self.embedding = nn.Linear(10, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, latent_size)

    def generate_positional_encoding(self, seq_len, hidden_size):
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0)/hidden_size))
        positional_encoding = torch.zeros(1, seq_len, hidden_size)
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return positional_encoding                                     ### (1, seq_len, hidden_size)

    def forward(self, x):
        x = x.reshape(-1, 750, 10)                                     
        x = self.embedding(x) + (self.position_encoding).to(x.device)  ### (batch_size, 750, hidden_size)
        x = x.permute(1, 0, 2)                                         ### (750, batch_size, hidden_size)
        encoded_output = self.transformer_encoder(x)
        encoded_output = encoded_output.permute(1, 0, 2)               ### (batch_size, 750, hidden_size)
        pooled_ouptput = torch.mean(encoded_output, dim=1)             ### (batch_size, hidden_size)
        out = self.fc(pooled_ouptput)                                  ### (batch_size, latent_size)

        return out



class xrd_encoder_CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_layers = 10
        self.conv_1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_9 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)

        for i in range(1, self.num_layers):
            self.add_module("pool_%d" % i, nn.MaxPool1d(kernel_size=2, stride=2))
        
        for j in range(1, self.num_layers):
            self.add_module("drop_%d" % j, nn.Dropout(0.2))
        
        self.fc = nn.Linear(256 * 14, 256)
        self.leakyrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # x (batch_size, 7500)
        x = x.unsqueeze(1)  # x (batch_size, 1, 7500)
        for i in range(1, self.num_layers):
            x = self._modules["conv_%d" % i](x)
            x = self.leakyrelu(x)
            x = self._modules["pool_%d" % i](x)
            x = self._modules["drop_%d" % i](x)

        x = self.flatten(x)
        x = self.fc(x)
        return x          # x (batch_size, 256)