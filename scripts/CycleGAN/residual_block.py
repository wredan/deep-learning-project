from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # Il blocco segue la struttura di un classico residual block di ResNet
        conv_block = [  nn.ReflectionPad2d(1), # Il ReflectionPad fa padding usando una versione "specchiata" del bordo dell'immagine
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features), # Instance normalization, un tipo di normalizzazione simile a batch normalization spesso usato per style transfer
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # La forward applica il blocco e somma l'input per ottenere una connessione residua
        return x + self.conv_block(x)