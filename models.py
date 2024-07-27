import torch 
from torch import nn 
import torchvision.models as models

class TinyVGG(nn.Module):
    """_summary_
    Here is architecture of TinyVGG: 
        https://poloclub.github.io/cnn-explainer/
        
    Args:
        input_shape: an integer indicating number of channels
        output_shape: an integer indicating number of output units (how much classes are)
        hidden_units: an integer indicating number of hidden units between layers.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.classifier(X)
        return X


def resnet_model(output_shape, device, pre_train_model=True):
    model = models.resnet50()
    if pre_train_model:
        pretrained_url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        # Download the weights
        state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=True)
        model.load_state_dict(state_dict)
    # Modify the fully connected layer to have 2 outputs
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_shape)
    model = model.to(device)
    return model
        