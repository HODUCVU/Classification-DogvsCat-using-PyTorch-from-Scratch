# Classification-DogvsCat-using-PyTorch-from-Scratch
🔰 This is a project that uses PyTorch to classify dogs and cats. I built it from scratch and compared multiple models to see which one worked better

⚡ **Notebook - Train models**
> Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCGTppY8ZcNOLiJ3ldsPp5qfXXdw3KOy?usp=sharing)

# 🔧 Deployment model
⚡ **Notebook - Evaluating models and deploy them**
> Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p4MuCCgUp1WsDdP5jg2Vb8wbPMsJfsqj?usp=sharing) 

📁 **File Structure**
```
demos/
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
## App demo ResNet50 model (deployed)

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
  
