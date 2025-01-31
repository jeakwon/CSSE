import torch

def selected_class_accuracy(model, dataloader, selected_classes, device):
    """Evaluate the model accuracy for selected classes."""
    model.eval()
    avg_acc = 0.0
    num_test_batches = 0

    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images = sample["image"].to(device)
            test_labels = sample["label"].to(device)  # Assumes one-hot encoded labels

            # Filter predictions and labels by selected classes
            mask = torch.isin(torch.argmax(test_labels, dim=1), torch.tensor(selected_classes, device=device))
            if not mask.any():
                continue

            images = images[mask]
            test_labels = test_labels[mask]

            test_predictions = model(images)
            predicted_labels = test_predictions.argmax(axis=1)
            true_labels = torch.argmax(test_labels, axis=1)

            # Compute accuracy for selected classes
            avg_acc += torch.mean((predicted_labels == true_labels).to(torch.float32))
            num_test_batches += 1
    acc = avg_acc / num_test_batches if num_test_batches > 0 else 0.0
    return acc.item()