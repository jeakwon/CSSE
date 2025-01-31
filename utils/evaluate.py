import torch

def selected_class_accuracy(model, dataloader, selected_classes, device):
    """Evaluate the model accuracy for selected classes."""
    model.eval()
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images = sample["image"].to(device)
            test_labels = sample["label"].to(device)  # Assumes one-hot encoded labels

            # Filter predictions and labels by selected classes
            mask = torch.isin(torch.argmax(test_labels, dim=1), torch.tensor(selected_classes, device=device))
            if not mask.any():
                continue  # Skip this batch if no selected class exists

            images = images[mask]
            test_labels = test_labels[mask]

            test_predictions = model(images)
            predicted_labels = test_predictions.argmax(axis=1)
            true_labels = torch.argmax(test_labels, axis=1)

            # Compute accuracy for selected classes
            correct_predictions = (predicted_labels == true_labels).sum().item()
            batch_size = len(true_labels)

            total_acc += correct_predictions
            total_samples += batch_size

    # Avoid division by zero
    if total_samples > 0 :
        acc = total_acc / total_samples
    else:
        acc = 0.0
        print('else')
    return acc
