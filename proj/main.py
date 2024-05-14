import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# load the training dataset

transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

def get_dataset_loader(train: bool, batch_size: int, params={}):
    return torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./dataset/', train=train, download=True, transform=transform),
                batch_size=batch_size, shuffle=True, **params)

# # show some images

# examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# fig.show()

# create neural network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# train the network

def train(epoch: int, network: Net, train_loader: torch.utils.data.DataLoader):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# test the model

def test(network: Net, test_loader: torch.utils.data.DataLoader):
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to('cuda'), target.to('cuda')
        output = network(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1000
    EPOCHS = 15

    network = Net().to('cuda')
    # optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    optimizer = torch.optim.Adadelta(network.parameters(), lr=1.0)

    train_loader = get_dataset_loader(train=True, batch_size=TRAIN_BATCH_SIZE, params={'num_workers': 1, 'pin_memory': True})

    for epoch in range(1, EPOCHS + 1):
        train(epoch, network, train_loader)

    # save the model
    torch.save(network.state_dict(), 'model.pth')
    print('Model saved')

    test_loader = get_dataset_loader(train=False, batch_size=TEST_BATCH_SIZE, params={'num_workers': 1, 'pin_memory': True})

    test(network, test_loader)