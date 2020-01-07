import ml_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thinc.api import xp2torch, get_shuffled_batches
import tqdm
import typer


class Net(nn.Module):
    def __init__(self, n_class, n_in, n_hidden, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.dropout1 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.dropout2 = nn.Dropout2d(0.2)
        self.fc3 = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=-1)
        return output


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({correct / len(test_loader.dataset):.0%}%)\n"
    )


def main(
    n_hidden: int = 32,
    dropout: float = 0.2,
    n_iter: int = 10,
    batch_size: int = 128,
    n_epoch: int = 10,
):
    torch.set_num_threads(1)
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    model = Net(10, 28 * 28, n_hidden)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(n_epoch):
        model.train()
        train_batches = list(get_shuffled_batches(train_X, train_Y, batch_size))
        for images, true_labels in tqdm.tqdm(train_batches):
            images = xp2torch(images)
            true_labels = xp2torch(true_labels)
            optimizer.zero_grad()
            guess_labels = model(images)
            loss = F.nll_loss(guess_labels, true_labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    typer.run(main)
