import torch 
import predictions
import os 
from torchvision import transforms
from data_setup import download_data, create_dataloader
from models import TinyVGG
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
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Create DataLoader
train_dataloader, valid_dataloade, cls_names = create_dataloader(train_dir, valid_dir, 
                                                                 transformer,
                                                                 BATCH_SIZE,
                                                                 NUM_WORKERS)
# Create model
model = TinyVGG(input_shape=3, 
                hidden_units=HIDDEN_UNITS, 
                output_shape=len(cls_names)).to(device)

# Start training 
results = train(model, train_dataloader, valid_dataloade, NUM_EPOCHS, LR, device)

# Save the model
save_model(model=model, tar_dir="models", model_name="10_mode_tinybgg.pth")

# plot the results
plot_training_and_testing_results(results)