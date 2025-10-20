import tqdm
from src.eval import evaluate
from src.utils import add_heads, update_memory
import torch
import random

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


def replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size,batch_size=64):
    model.train()
    running_loss = 0.0
    memory_dataset = torch.utils.data.TensorDataset(
        torch.stack(memory_set["data"]), 
        torch.tensor(memory_set["labels"])
        )
    combined_dataset = torch.utils.data.ConcatDataset([train_set, memory_dataset])
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss = running_loss / (progress_bar.n + 1))
        epoch_loss = running_loss / len(combined_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    # Update memory set after training
    new_data = []
    new_labels = []
    categories = len(set(memory_set["labels"]))
    if categories == 0:
        indices = random.sample(range(len(train_set)), memory_size)
    else:
        indices = random.sample(range(len(train_set)), memory_size//categories)
    for idx in indices:
        data, label = train_set[idx]
        new_data.append(data.cpu())
        new_labels.append(label)
    memory_set = update_memory(memory_set, new_data, new_labels, memory_size)
    
