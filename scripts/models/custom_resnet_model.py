from torch import nn

class CustomResNetModule(nn.Module):
    def __init__(self, resnet_model, num_classes):
        super().__init__()

        self.in_features = resnet_model.fc.in_features

        resnet_model._modules.pop(list(resnet_model._modules.keys())[-1]) # pop fully connected layer
        # feature extractor from resnet
        self.feature_extractor = nn.Sequential(resnet_model._modules)

        # classifier from resnet
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # features extraction and reshaping to 1-dim array
        features = self.feature_extractor(x).view(x.shape[0], -1)
        return self.classifier(features)
