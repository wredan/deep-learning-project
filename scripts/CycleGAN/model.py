import pytorch_lightning as pl
import itertools
import torchvision
from scripts.CycleGAN.generator import Generator
from scripts.CycleGAN.discriminator import Discriminator
from scripts.CycleGAN.utils import weights_init_normal, LambdaLR
import torch
class CycleGAN(pl.LightningModule):

    A2B = 0
    B2A = 1

    def __init__(self, 
                 input_nc = 3, 
                 output_nc = 3, 
                 lr=0.0002, 
                 betas=(0.5, 0.999), 
                 starting_epoch=0, 
                 n_epochs=200, 
                 decay_epoch=100,
                 print_images_each_N_batch = 500
                  ):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        
        # Generators A2B, B2A
        self.netG_A2B = Generator(input_nc, output_nc)
        self.netG_B2A = Generator(output_nc, input_nc)
        
        # A and B Discriminator
        self.netD_A = Discriminator(input_nc)
        self.netD_B = Discriminator(output_nc)
        
        # Normalization
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        
        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
                    
    def forward(self, x, mode=A2B):
        if mode== CycleGAN.A2B:
            return self.netG_A2B(x)
        else:
            return self.netG_B2A(x)
    
    def configure_optimizers(self):
        # Generator Optimizer
        optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                lr=self.hparams.lr, betas=self.hparams.betas)
        
        # Discriminator Optimizers
        optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        
        # Scheduler learning rate decay
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        
        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch
        
        target_real = torch.ones((real_A.shape[0],1)).type_as(real_A)
        target_fake = torch.zeros((real_A.shape[0],1)).type_as(real_A)
        
        # Generator
        if optimizer_idx==0:
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.netG_A2B(real_B)
            loss_identity_B = self.criterion_identity(same_B, real_B)*5.0

            # G_B2A(A) should equal A if real A is fed
            same_A = self.netG_B2A(real_A)
            loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = self.netG_A2B(real_A) 
            pred_fake = self.netD_B(fake_B) # discriminator B prediction
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real) # loss GAN A2B

            fake_A = self.netG_B2A(real_B)
            pred_fake = self.netD_A(fake_A)  # discriminator A prediction
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real) # loss GAN B2A

            # Cycle consistency loss
            recovered_A = self.netG_B2A(fake_B) #fake B -> B2A - should be same as real_A
            loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0 #cycle consistency loss

            recovered_B = self.netG_A2B(fake_A) #fake A -> A2B - should be same as real_B
            loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10.0 #cycle consistency loss

            # Overall loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            self.log('loss_G/loss_identity_A', loss_identity_A, on_epoch=True)
            self.log('loss_G/loss_identity_B', loss_identity_B, on_epoch=True)
            self.log('loss_G/loss_GAN_A2B', loss_GAN_A2B, on_epoch=True)
            self.log('loss_G/loss_GAN_B2A', loss_GAN_B2A, on_epoch=True)
            self.log('loss_G/loss_cycle_ABA', loss_cycle_ABA, on_epoch=True)
            self.log('loss_G/loss_cycle_BAB', loss_cycle_BAB, on_epoch=True)
            self.log('loss_G/overall', loss_G, on_epoch=True)
            
            # Log images to visually check the trainig
            if batch_idx % self.hparams.print_images_each_N_batch == 0:
                grid_A = torchvision.utils.make_grid(real_A[:50], nrow=10, normalize=True)
                grid_A2B = torchvision.utils.make_grid(fake_B[:50], nrow=10, normalize=True)
                grid_A2B2A = torchvision.utils.make_grid(recovered_A[:50], nrow=10, normalize=True)
                
                grid_B = torchvision.utils.make_grid(real_B[:50], nrow=10, normalize=True)
                grid_B2A = torchvision.utils.make_grid(fake_A[:50], nrow=10, normalize=True)
                grid_B2A2B = torchvision.utils.make_grid(recovered_B[:50], nrow=10, normalize=True)
                
                self.logger.experiment.add_image('A/A', grid_A, self.global_step)
                self.logger.experiment.add_image('A/A2B', grid_A2B, self.global_step)
                self.logger.experiment.add_image('A/A2B2A', grid_A2B2A, self.global_step)
                
                self.logger.experiment.add_image('B/B', grid_B, self.global_step)
                self.logger.experiment.add_image('B/B2A', grid_B2A, self.global_step)
                self.logger.experiment.add_image('B/B2A2B', grid_B2A2B, self.global_step)
                
            return loss_G
        
        # Discriminator A
        elif optimizer_idx==1:
            # Real loss
            pred_real = self.netD_A(real_A)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = self.netG_B2A(real_B)
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # global loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            self.log('loss_D/loss_D_A',loss_D_A, on_epoch=True)
            return loss_D_A
        
        # Discriminator B
        elif optimizer_idx==2:
            pred_real = self.netD_B(real_B)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = self.netG_A2B(real_A) 
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # global loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            self.log('loss_D/loss_D_B', loss_D_B, on_epoch=True)
            return loss_D_B