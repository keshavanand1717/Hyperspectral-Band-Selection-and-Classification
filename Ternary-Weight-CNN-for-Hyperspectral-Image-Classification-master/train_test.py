import torch
import torch.optim as optim

def train_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, epochs: int = 1) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    print("Training...")
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0

        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)

            outputs = model(samples)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            logits, *extras = outputs
            predicted = logits.argmax(dim=1)
            correct = (predicted == labels).sum()
            total_correct += correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_samples = len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/total_samples:.4f}, Accuracy={100 * total_correct/total_samples:.2f}%")


def test_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0

    print("Testing...")
    with torch.no_grad():
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)

            outputs = model(samples)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            logits, *extras = outputs
            predicted = logits.argmax(dim=1)
            correct = (predicted == labels).sum()
            total_correct += correct.item()

    total_samples = len(dataloader.dataset)
    print(f"Loss={total_loss/total_samples:.4f}, Accuracy={100 * total_correct/total_samples:.2f}%")
