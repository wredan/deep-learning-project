class LambdaLR(): # leaning rate decay
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # Inizializziamo le convoluzioni con rumore gaussiano
    # di media zero e deviazione standard 0.02
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # Nel caso della batchnorm2d useremo media 1 e deviazione standard 0.02
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        # il bias è costante e pari a zero
        m.bias.data.fill_(0.0)