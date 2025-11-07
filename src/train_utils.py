import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_index, log_interval=10):
    model.train()
    running_loss = 0.0

    for batch_index, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % log_interval == (log_interval - 1):
            avg_loss = running_loss / log_interval
            print(f"Epoch: {epoch_index + 1}, batch: {batch_index}, loss: {avg_loss:.4f}")
            running_loss = 0.0

def evaluate_loss_acc(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, y_true, y_pred