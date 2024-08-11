# Classification-DogvsCat-using-PyTorch-from-Scratch
🔰 This is a project that uses PyTorch to classify dogs and cats. I built it from scratch and compared multiple models to see which one worked better

📎 I do this project to practice what I learned from this course [PyTorch for Deep Learning & Machine Learning – Full Course with 30 hours - freecodecamp.org](https://www.youtube.com/watch?v=V_xro1bcAuA&list=LL&index=15&t=92s)

⚡ **Notebook - Train models**
> Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCGTppY8ZcNOLiJ3ldsPp5qfXXdw3KOy?usp=sharing)

🖱️ If you don't want to train on Colab, well, just run this statement below:
```
User> python3 train_resnet_model.py
```
📉 Evaluating Train and Testing Processing
- We can see that the results from the ResNet50 model are much better than the TinyVGG model, however, in the ResNet50 model, we can see that the model is overfitting. To solve this problem, we can reduce the number of layers in the ResNet50 model, in addition, we can experiment with some other optimizer types such as SGD or Adam to see the efficiency.
  
    <img width="600" alt="TinyVGG model" src="https://github.com/user-attachments/assets/3f83820b-9582-434f-8692-bec7dcbe1af3">
    <img width="600" alt="ResNet50 model" src="https://github.com/user-attachments/assets/a218c5b1-6a15-4995-adb4-687ebb2cc1b6">

😾 Predict with TinyVGG model

<img width="350" height="350" alt="Dog with TinyVGG model" src="https://github.com/user-attachments/assets/72888ad3-336b-44cc-99ec-ea71a5d26d56">
<img width="400" height="350" alt="Cat with TinyVGG model" src="https://github.com/user-attachments/assets/c40a8229-07d9-4ec9-98fb-d1054d7b8be7">


😾 Predict with ResNet50 model

<img width="350" height="350" alt="Cat with ResNet50 model" src="https://github.com/user-attachments/assets/f0890b4a-827a-40fa-af3c-12233cb44527">
<img width="400" height="350" alt="Dog with ResNet50 model" src="https://github.com/user-attachments/assets/0fc42aac-01d8-4bf5-b5c7-ffdcab7b0c76">



# Evaluating models and deploy them
⚡ **Notebook - Evaluating models and deploy them**
> Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p4MuCCgUp1WsDdP5jg2Vb8wbPMsJfsqj?usp=sharing) 

📁 **File Structure**
```
deploy/
└── dogvscat_mini/
    ├── ResNet.pth
    ├── app.py
    ├── examples/
    │   ├── example_1.jpg
    │   ├── example_2.jpg
    │   └── example_3.jpg
    ├── model.py
    └── requirements.txt
```
### Using git tokens to push on huggingface space
- See more here https://dev.to/sh20raj/how-to-use-git-with-hugging-face-from-cloning-to-pushing-with-access-token-verification-5711
- format: 
    ```
      git clone https://USERNAME:YOUR_ACCESS_TOKEN@huggingface.co/spaces/USERNAME/REPO_NAME.git
    ```
- or:  
    ```
      git remote set-url origin https://USERNAME:YOUR_ACCESS_TOKEN@huggingface.co/spaces/USERNAME/REPO_NAME.git
    ```
  
