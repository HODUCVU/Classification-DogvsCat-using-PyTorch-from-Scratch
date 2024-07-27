import torch
from tqdm import tqdm 
from timeit import default_timer as timer 

def train_step(model, train_dataloader, optimizer, loss_fn, device):
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y ) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # Forward pass 
        y_pred = model(X)
        # Compute loss and accuracy 
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        correct = torch.eq(torch.argmax(torch.softmax(y_pred, dim=1), dim=1), y).sum().item()
        train_acc += (correct)/len(y) 

        # Zero gradients and backward pass 
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
    
    return train_loss / len(train_dataloader), train_acc / len(train_dataloader)

def validate_step(model, valid_dataloader, loss_fn, device):
    model.eval()
    valid_loss, valid_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.item()
            valid_acc += torch.eq(torch.argmax(torch.softmax(y_pred, dim=1), dim=1), y).sum().item()
        
        valid_loss = valid_loss / len(valid_dataloader)
        valid_acc = valid_acc / len(valid_dataloader)
    return valid_loss, valid_acc

def train(model, train_dataloader, valid_dataloader, epochs, lr, device):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'time_train': 0,
        'model_name': model.__class__.__name__,
        'device': device
    }
    start_time = timer() 
    for epoch in tqdm(range(epochs)):
        print("\nEpoch {epoch}")
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, device)
        valid_loss, valid_acc = validate_step(model, valid_dataloader, loss_fn, device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc)
        if epoch % 5 == 0:
            print(f"Train loss {train_loss} | Train acc {train_acc} | Valid loss {valid_loss} | Valid acc {valid_acc}")
    end_time = timer()
    results['time_train'] = end_time - start_time
    print(f'\nTime train {model.__class__.__name__} on {device} is {end_time - start_time} seconds')
    return results