
def create_resnet_model(num_classes:int=2,
                        seed:int=42):
  torch.manual_seed(seed)
  # Load the ResNet-50 model
  model_resnet = models.resnet50()
  # Modify the fully connected layer to have 2 outputs
  num_ftrs = model_resnet.fc.in_features
  model_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
  model_resnet = model_resnet.to(device) 
  transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.CenterCrop(size=(224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return model_resnet, transforms
