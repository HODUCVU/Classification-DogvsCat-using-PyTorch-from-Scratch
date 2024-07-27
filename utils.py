import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct *100)/len(y_true)
    return acc

def save_model(model:torch.nn.Module, tar_dir:str, model_name:str):
    # Create target directory
    target_dir = Path(tar_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model save path 
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_save_path = target_dir/model_name 
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    
def plot_training_and_testing_results(results):
    train_loss = results['train_loss']
    train_acc = results['train_acc']
    test_loss = results['test_loss']
    test_acc = results['test_acc']

    epochs = range(len(train_loss))

    plt.figure(figsize=(10,7))
    plt.suptitle(f'Time train {results["model_name"]} on {results["device"]} in {epochs[-1] + 1} epochs is {results["time_train"]:.3f} seconds', fontsize=15)

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label='train_acc')
    plt.plot(epochs, test_acc, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Compare models
def compare_models(results_model_0, results_model_1):
    compare_model = pd.DataFrame({'model_0': {"Model name": results_model_0["model_name"],
                                        "Train loss": results_model_0['train_loss'][-1],
                                        "Train Accuracy": results_model_0['train_acc'][-1],
                                        "Test loss": results_model_0['test_loss'][-1],
                                        "Test Accuracy": results_model_0['test_acc'][-1],
                                        "Time training": results_model_0['time_train']},
                            'model_resnet': {"Model name": results_model_1["model_name"],
                                        "Train loss": results_model_1['train_loss'][-1],
                                        "Train Accuracy": results_model_1['train_acc'][-1],
                                        "Test loss": results_model_1['test_loss'][-1],
                                        "Test Accuracy": results_model_1['test_acc'][-1],
                                        "Time training": results_model_1['time_train']}})
    return compare_model