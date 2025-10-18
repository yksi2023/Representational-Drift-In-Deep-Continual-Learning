import tqdm
from src.eval import evaluate
from src.utils import add_heads
import torch

def normal_train(model, train_loader, criterion, optimizer, device, epochs, val_loader=None):
    model.train()
    running_loss = 0.0
    
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss = running_loss / (progress_bar.n + 1))
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")


