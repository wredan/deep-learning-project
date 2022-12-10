from torch import nn
import torchmetrics
import torch
import pytorch_lightning as pl
import numpy as np
class CustomResNetModule(pl.LightningModule):
    def __init__(self, resnet_model, num_classes, lr=1e-3):
        super(CustomResNetModule, self).__init__()
        self.save_hyperparameters(ignore=['resnet_model'])

        self.in_features = resnet_model.fc.in_features

        # pop fully connected layer
        resnet_model._modules.pop(list(resnet_model._modules.keys())[-1])

        # feature extractor from resnet
        self.feature_extractor = nn.Sequential(resnet_model._modules)

        # classifier from resnet
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.Dropout(0.5),
            nn.Linear(512, self.hparams.num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # features extraction and reshaping to 1-dim array
        features = self.feature_extractor(x).view(x.shape[0], -1)
        return self.classifier(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):        
        x, y = batch        
        preds = self.forward(x)        
        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)
        
        self.log('training/loss', loss.item(), on_epoch=True)
        self.log('training/accuracy', self.train_acc, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):        
        x,y = batch        
        preds = self.forward(x)        
        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)
        
        self.log('validation/loss', loss.item(), on_epoch=True)
        self.log('validation/accuracy', self.val_acc, on_epoch=True)
        
    def test_step(self, batch, batch_idx):        
        x,y = batch
        preds = self.forward(x)

        self.test_acc(torch.argmax(preds, dim=1), y)        
        self.log('test/accuracy', self.test_acc, on_epoch=True)
        