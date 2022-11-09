from torch import nn
from .gradient_reversal import revgrad

class DomainDiscriminatorGRL(nn.Module):
    def __init__(self):
        super(DomainDiscriminatorGRL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
    )

    def forward(self, x, lambda_=1):
        return self.model(revgrad(x, lambda_))