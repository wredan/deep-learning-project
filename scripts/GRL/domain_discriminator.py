from torch import nn
from .gradient_reversal import revgrad

class DomainDiscriminatorGRL(nn.Module):
    def __init__(self, in_features):
        super(DomainDiscriminatorGRL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
    )

    def forward(self, x, lambda_=1):
        return self.model(revgrad(x, lambda_))