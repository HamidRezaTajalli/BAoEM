import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils import resample
from tqdm import tqdm




def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_corrects = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * len(data)
        pred = torch.max(output, dim=1)
        corrects = torch.sum(pred[1] == target).item()
        total_corrects += corrects
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_corrects / len(train_loader.dataset)

    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_corrects = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)

            pred = torch.max(output, dim=1)
            corrects = torch.sum(pred[1] == target).item()
            total_corrects += corrects

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = total_corrects / len(test_loader.dataset)
    return avg_loss, accuracy

def test_ensemble_bagging(models_list, device, test_loader, criterion):
    for model in models_list:
        model.eval()

    total_corrects = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(len(target), len(models_list))  # A matrix to store each model's prediction for each input

            # Get each model's prediction for each input batch
            for model_idx, model in enumerate(models_list):
                output = model(data)
                outputs[:, model_idx] = torch.max(output, dim=1)[1]

            # Use a majority vote system to get final predictions
            final_predictions, _ = torch.mode(outputs, dim=1)
            final_predictions = final_predictions.to(device)
            corrects = torch.sum(final_predictions == target).item()
            total_corrects += corrects

    accuracy = total_corrects / len(test_loader.dataset)
    return accuracy
