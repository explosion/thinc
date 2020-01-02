from __future__ import print_function
import ml_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thinc.util import xp2torch, get_shuffled_batches
import tqdm


class Net(nn.Module):
    def __init__(self, n_class, n_in, n_hidden, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.dropout1 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.dropout2 = nn.Dropout2d(0.2)
        self.fc3 = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=-1)
        return output


def load_mnist():
    from thinc.backends import NumpyOps
    from thinc.util import to_categorical
    ops = NumpyOps()
    mnist_train, mnist_dev, _ = ml_datasets.mnist()
    train_X, train_Y = ops.unzip(mnist_train)
    dev_X, dev_Y = ops.unzip(mnist_dev)
    train_Y = train_Y.astype("int64")
    dev_Y = dev_Y.astype("int64")
    return (train_X, train_Y), (dev_X, dev_Y)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(n_hidden=32, dropout=0.2, n_iter=10, batch_size=128, n_epoch=10):
    torch.set_num_threads(1)
    (train_X, train_Y), (dev_X, dev_Y) = load_mnist()
    model = Net(10, 28*28, n_hidden)
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


if __name__ == '__main__':
    import plac
    plac.call(main)
