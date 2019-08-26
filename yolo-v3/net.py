import torch.nn as nn

class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(152, 80),
            nn.Linear(80, 32),
            nn
        )

    def forward(self, x):
        return detect_13, detect_26, detect_52