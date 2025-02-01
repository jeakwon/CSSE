import torch
import torch.nn as nn
import torch.optim as optim
from csse.utils.evaluate import selected_class_accuracy

def unlearn_one_epoch(forget_classes, model, data_loader, criterion, optimizer, device='cuda'):
    model.train()
    avg_loss = 0.0

    entire_classes = torch.tensor( data_loader.dataset.classes, device=device)
    forget_classes = torch.tensor( forget_classes, device=device)
    forget_indices = torch.where( torch.isin(entire_classes, forget_classes) )[0]

    assert torch.isin( forget_classes, entire_classes ).all().item(), f'Selected classes must be included in data_loader partition'
    assert len(data_loader)>0, 'Data loader is empty'

    for _, sample in enumerate(data_loader):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)  # one-hot encoded labels

        # Set selected class labels to chance level (1/num_classes)
        mask = torch.isin(labels.argmax(dim=1), forget_indices)
        labels[mask] = torch.ones_like(labels[mask]) / len(entire_classes)

        # Rearrange model outputs to match data loader class setting.
        outputs = model(images) # (batch_size x 100)
        outputs = outputs[:, entire_classes] # (batch_size x num_classes) & change order

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    avg_loss /= len(data_loader)
    return avg_loss

def unlearning_accuacy(forget_classes, model, data_loader, device='cuda'):
    entire_classes = torch.tensor( data_loader.dataset.classes, device=device)
    forget_classes = torch.tensor( forget_classes, device=device)
    retain_classes = torch.tensor( [c for c in entire_classes if c not in forget_classes], device=device)

    forget_acc = selected_class_accuracy(model, data_loader, forget_classes, device='cuda')
    retain_acc = selected_class_accuracy(model, data_loader, retain_classes, device='cuda')
    rfa = retain_acc - forget_acc
    return retain_acc, forget_acc, rfa

class Unlearning:
    def __init__(self, model, train_loader, valid_loader, test_loader, criterion, optimizer, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def run(self, forget_classes, epochs=50, save_path='./best_model.pt', early_stop_patience=5):
        baseline_retain_acc, baseline_forget_acc, baseline_rfa = unlearning_accuacy(forget_classes, self.model, self.valid_loader, device=self.device)

        print(f"**Training** Loss <- Trainset | Retain Acc. <- Validset | Forget Acc. <- Validset | RFA <- Validset")
        print(f"[ Baseline ] Loss: {'N/A':>10} | Retain Acc.: {baseline_retain_acc:10.2%} | Forget Acc.: {baseline_forget_acc:10.2%} | RFA: {baseline_rfa:10.2%}")

        best_rfa = baseline_rfa
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, epochs+1):
            loss = unlearn_one_epoch(forget_classes, self.model, self.train_loader, self.criterion, self.optimizer, device=self.device)
            
            retain_acc, forget_acc, rfa = unlearning_accuacy(forget_classes, self.model, self.valid_loader, device=self.device
                                                             )
            if rfa > best_rfa:
                best_rfa = rfa
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                epochs_without_improvement = 0
                mark_best = f' <-- BEST (Saved as {save_path})'
            else:
                mark_best = ''
                epochs_without_improvement += 1

            print(f"[ Epoch {epoch:2d} ] Loss: {loss:10.4f} | Retain Acc.: {retain_acc:10.2%} | Forget Acc.: {forget_acc:10.2%} | RFA: {rfa:10.2%}{mark_best}")

            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping triggered. No improvement in RFA for {early_stop_patience} consecutive epochs.")
                break


        print("\nTraining finished. Loading best model for testset evaluation...")
        
        self.model.load_state_dict(torch.load(save_path, weights_only=True))
        endpoint_retain_acc, endpoint_forget_acc, endpoint_rfa = unlearning_accuacy(forget_classes, self.model, self.test_loader, device=self.device)
        print(f"**Testing*** Loss <- Trainset  | Retain Acc. <- Testset  | Forget Acc. <- Testset  | RFA <- Testset ")
        print(f"[ Endpoint ] Loss: {'N/A':>10} | Retain Acc.: {endpoint_retain_acc:10.2%} | Forget Acc.: {endpoint_forget_acc:10.2%} | RFA: {endpoint_rfa:10.2%}")

        result = dict(
            best_epoch=best_epoch,
            best_rfa=best_rfa,
            baseline_retain_acc=baseline_retain_acc, 
            baseline_forget_acc=baseline_forget_acc, 
            baseline_rfa=baseline_rfa,
            endpoint_retain_acc=endpoint_retain_acc, 
            endpoint_forget_acc=endpoint_forget_acc, 
            endpoint_rfa=endpoint_rfa,
        )
        return self.model, result
