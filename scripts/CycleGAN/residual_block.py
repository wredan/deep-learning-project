from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # The block follows the structure of a classic ResNet residual block
        conv_block = [  nn.ReflectionPad2d(1), # ReflectionPad pads using a "mirrored" version of the edge of the image
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features), # Instance normalization, a type of normalization similar to batch normalization often used for style transfer
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # The forward applies the block and sums the input to obtain a residual connection
        return x + self.conv_block(x)