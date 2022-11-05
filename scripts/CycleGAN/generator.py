from torch import nn

from .residual_block import ResidualBlock

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Blocco di convoluzioni iniziale che mappa l'input su 64 feature maps
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # ============ Encoder ============
        # Due blocchi che mappano l'input
        # da 64 a 128 e da 128 a 256 mappe
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Aggiungiamo dunque i residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # ============ Decoder =============
        # Due blocchi di convoluzione
        out_features = in_features//2
        for _ in range(2):
            # Qui usiamo la convoluzione trasposta (https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
            # che fa upsampling piuttosto che downsampling quando si impostano stride e padding
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Layer finale di output
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        # Inizializziamo l'oggetto sequential con la lista dei moduli
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)