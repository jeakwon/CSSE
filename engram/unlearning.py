import torch

def unlearn_one_epoch(selected_classes, model, data_loader, criterion, optimizer, device='cuda'):
    model.train()
    avg_loss = 0.0

    retained_classes = torch.tensor( data_loader.dataset.classes, device=device)
    selected_classes = torch.tensor( selected_classes, device=device)
    selected_class_indices = torch.where( torch.isin(retained_classes, selected_classes) )[0]

    assert torch.isin( selected_classes, retained_classes ).all().item(), f'Selected classes must be included in data_loader partition'

    if len(data_loader)==0:
        return 0.0

    for _, sample in enumerate(data_loader):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)  # one-hot encoded labels

        # Set selected class labels to chance level (1/num_classes)
        mask = torch.isin(labels.argmax(dim=1), selected_class_indices)
        labels[mask] = torch.ones_like(labels[mask]) / len(retained_classes)

        # Rearrange outputs to follow number and order of retained_classes
        outputs = model(images)[:, retained_classes]

        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    avg_loss /= len(data_loader)
    return avg_loss