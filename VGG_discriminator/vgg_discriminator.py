import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import classifier_lib

# Define a custom classifier for your task
class vgg_discriminator(nn.Module):
    def __init__(self, num_classes):
        super(vgg_discriminator, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # Adjust num_classes as needed
            nn.Sigmoid()  # Use sigmoid for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = self.classifier(x)
        return x

def get_new_discriminator(discriminator_ckpt):
    model = vgg_discriminator(num_classes=1)  # Adjust num_classes as needed

    # Load the model weights from the saved file
    model.load_state_dict(torch.load(discriminator_ckpt))

    # Set the model to evaluation mode (if needed)
    model.eval()

    return model