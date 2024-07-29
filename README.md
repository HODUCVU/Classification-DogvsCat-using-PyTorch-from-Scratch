# Classification-DogvsCat-using-PyTorch-from-Scratch
This is a project that uses PyTorch to classify dogs and cats. I built it from scratch and compared multiple models to see which one worked better

Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCGTppY8ZcNOLiJ3ldsPp5qfXXdw3KOy?usp=sharing)

# Compare our models

# Deployment model with Gradio on Huggingface

Open with [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p4MuCCgUp1WsDdP5jg2Vb8wbPMsJfsqj?usp=sharing)

```
demos/
└── foodvision_mini/
    ├── 09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth
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
- Example: 
    ```
        git remote add origin https://hoducvu1234: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@huggingface.co/spaces/Duc-Vu/Dogs-and-Cats
    ```

  
