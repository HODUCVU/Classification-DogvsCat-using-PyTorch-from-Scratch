import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt

def download_image(url, file_name):
    print(f'Downloading {file_name}...')
    request = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(request.content)
        return Image.open(file_name)

def make_predict(model, classes, image, transformer, device):
    model.eval()
    transformed_image = transformer(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_pred = model(transformed_image).to(device)
        prob = torch.softmax(y_pred, dim=1).cpu().max().item()
        cls_to_idx = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        class_name = classes[cls_to_idx]
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.title(f'class: {class_name} | prob: {prob:.3f}', fontsize=15)
        plt.axis(False)