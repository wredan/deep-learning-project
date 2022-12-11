from torch import nn

from .residual_block import ResidualBlock

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block that maps the input to 64 feature maps
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # ============ Encoder ============
        # Two blocks that map the input
        # from 64 to 128 and from 128 to 256 feature maps
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Add the residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # ============ Decoder =============
        # Two convolution blocks
        out_features = in_features//2
        for _ in range(2):
            # Here we use transposed convolution (https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
            # that does upsampling rather than downsampling when setting stride and padding
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output final layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        # Initialize the sequential object with the list of modules
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)