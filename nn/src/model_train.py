import torch
import torch.nn as nn


def train_fn(model, optimizer, criterion, dataloader, device, sum_loss=True):
    model.train()
    final_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if sum_loss:
            loss += nn.L1Loss()(torch.sum(targets, 1), torch.sum(outputs, 1)) * 8e-7
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    final_loss /= len(dataloader)
    return final_loss


def valid_fn(model, criterion, inputs, targets, device):
    model.eval()
    inputs = torch.tensor(inputs.values, dtype=torch.float).to(device)
    targets = torch.tensor(targets.values, dtype=torch.float).to(device)
    targets = torch.reshape(targets, (-1, 1))
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()
    return loss


def inference_fn(model, inputs, device):
    model.eval()
    inputs = torch.tensor(inputs.values, dtype=torch.float).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.sigmoid().detach().cpu().numpy()

