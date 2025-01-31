import torch

def selected_class_accuracy(model, dataloader, selected_classes, device):
    """Evaluate the model accuracy for selected classes."""
    model.eval()
    total_acc = 0.0
    total_samples = 0

    class_partition = dataloader.dataset.classes
    is_selected_classes_in_partition = torch.isin(
        torch.tensor(selected_classes), torch.tensor(class_partition)
        ).all().item()
    assert is_selected_classes_in_partition, f'Selected classes must be included in dataloaders partition'

    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images = sample["image"].to(device)
            labels = sample["label"].to(device)  # Assumes one-hot encoded labels

            # Filter predictions and labels by selected classes
            mask = torch.isin(labels.argmax(dim=1), torch.tensor(selected_classes, device=device))
            if not mask.any():
                continue  # Skip this batch if no selected class exists
            
            images = images[mask]
            labels = labels[mask]

            pred = model(images)[:, class_partition] # Matchs with dataloader labels
            pred_labels = pred.argmax(dim=1)
            true_labels = labels.argmax(dim=1)

            # Compute accuracy for selected classes
            correct_predictions = (pred_labels == true_labels).sum().item()
            batch_size = len(true_labels)

            total_acc += correct_predictions
            total_samples += batch_size

    acc = total_acc / total_samples if total_samples > 0 else 0.0
    return acc
