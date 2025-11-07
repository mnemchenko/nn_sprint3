import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MedsClassifier(nn.Module):
    def __init__(self, num_classes=84):
        super(MedsClassifier, self).__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.backbone = mobilenet_v3_large(weights=weights)

        in_features = self.backbone.classifier[3].in_features

        self.backbone.classifier[3] = nn.Linear(in_features, num_classes, bias=True)

        for param in self.backbone.features.parameters():
            param.requires_grad = False

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)