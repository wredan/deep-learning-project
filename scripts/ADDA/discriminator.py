from torch import nn

class Discriminator(nn.Module):

    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.model(x)