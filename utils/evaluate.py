import torch

def selected_class_accuracy(model, data_loader, selected_classes, device, verbose=False):
    """Evaluate the model accuracy for selected classes."""
    model.eval()
    total_acc = 0.0
    total_samples = 0

    existing_classes = torch.as_tensor(data_loader.dataset.classes).to(device)
    selected_classes = torch.as_tensor(selected_classes).to(device)
    selected_class_indices = torch.where( torch.isin(existing_classes, selected_classes) )[0]
            
    
    assert torch.isin( selected_classes, existing_classes ).all().item(), f'Selected classes must be included in dataloaders partition'

    if len(data_loader)==0:
        return 0.0
        
    with torch.no_grad():
        for _, sample in enumerate(data_loader):
            images = sample["image"].to(device)
            labels = sample["label"].to(device)  # Assumes one-hot encoded labels

            # Filter predictions and labels by selected classes
            mask = torch.isin(labels.argmax(dim=1), selected_class_indices)
            if not mask.any():
                continue  # Skip this batch if no selected class exists
            
            images = images[mask]
            labels = labels[mask]

            pred = model(images)[:, existing_classes] # Matchs with data_loader label order
            pred_labels = pred.argmax(dim=1)
            true_labels = labels.argmax(dim=1)

            # Compute accuracy for selected classes
            correct_predictions = (pred_labels == true_labels).sum().item()
            batch_size = len(true_labels)

            total_acc += correct_predictions
            total_samples += batch_size

    acc = total_acc / total_samples if total_samples > 0 else 0.0
    return acc

