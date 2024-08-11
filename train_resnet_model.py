import torch 
import predictions
import os 
from torchvision import transforms
from data_setup import download_data, create_dataloader
from models import resnet_model
from engine import train
from utils import save_model, plot_training_and_testing_results
# Setup huperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
NUM_WORKERS = 0
LR = 0.001

# Setup directories
train_dir, valid_dir = download_data(root_path='./data', zipfile_name='dogvscat.zip')

# Setup target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transformer = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.CenterCrop(size=(224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoader
train_dataloader, valid_dataloade, cls_names = create_dataloader(train_dir, valid_dir, 
                                                                 transformer,
                                                                 BATCH_SIZE,
                                                                 NUM_WORKERS)
# Create model
model_resnet = resnet_model(output_shape=2, 
                     device=device, 
                     pre_train_model=True).to(device)

# Start training 
results_restnet = train(model_resnet, train_dataloader, valid_dataloade, NUM_EPOCHS, LR, device)

# Save the model
save_model(model=model_resnet, tar_dir="models", model_name="ResNet.pth")

# plot the results
plot_training_and_testing_results(results_restnet)