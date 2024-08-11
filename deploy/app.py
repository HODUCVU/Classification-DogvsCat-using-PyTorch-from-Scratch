
import gradio as gr
import os 
import torch

from model import create_resnet_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['dog', 'cat']

# Create model
resnet50, resnet50_transforms = create_resnet_model(num_classes=2,
                                                    seed=42)

# load saved weights
resnet50.load_state_dict(torch.load(f="ResNet.pth"),
                         map_location=torch.device("cpu"))

# Predict function
def predict(img):

  start_time = timer()

  img = resnet50_transforms(img).unsqueeze(0)

  resnet50.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(resnet50(img), dim=1)
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_name))}

  pred_time = round(timer() - start_time, 4)
  return pred_labels_and_probs, pred_time

# Gradio app
# Create title, description and article strings
title = "Dogs and Cats Mini"
title = title + "   ≽^•⩊•^≼   " + " ♡  " + "   ૮₍´｡ᵔ ꈊ ᵔ｡`₎ა    "

description = "An ResNet50 feature extractor computer vision model to classify images of animals as dog and cat."

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=2, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    # Create examples list
                    examples=example_list,
                    title=title,
                    descriptioon=description)

# Launch the demo!
demo.launch()
