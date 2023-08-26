import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, embedding, constant_embedding):
        super().__init__()
        self.embedding = embedding
        self.constant_embedding = constant_embedding
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.af = nn.ReLU()
        self.conv1 = nn.Conv1d(2 * 768, 100, 3)
        self.conv2 = nn.Conv1d(2 * 768, 100, 4)
        self.conv3 = nn.Conv1d(2 * 768, 100, 5)
        self.fc = nn.Linear(100 * 3, 2)

    def forward(self):
        embedding = torch.cat((self.embedding, self.constant_embedding), dim=2)
        embedding = embedding.permute(0, 2, 1)
        x = torch.squeeze(self.af(self.pool(self.conv1(embedding))), dim=-1)

        y = torch.squeeze(self.af(self.pool(self.conv2(embedding))), dim=-1)

        z = torch.squeeze(self.af(self.pool(self.conv3(embedding))), dim=-1)

        core = torch.cat((x, y, z), dim=1)
        output = self.fc(self.dropout(core))

        return output
