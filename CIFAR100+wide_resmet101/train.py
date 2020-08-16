import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from .datasets import train_dataset, test_dataset, train_loader, test_loader
from .models import net

ROOT = '/home/intern/'
# ROOT = '/drive/My drive/Colab Notebooks/CIFAR-try diff impl'
device = "cuda"
gc.collect()

# Displaying info about number of examples, iterations
classes = train_dataset.classes
num_classes = len(classes)
print("Num classes:", num_classes)
print("Number of train examples:", len(train_dataset))
print("Batch size:", 1024)
print("Number of iterations:", int(len(train_dataset) / 1024))

print("Number of test examples:", len(test_dataset))

# Visualisationg datsets
f = plt.figure()
fig, ax = plt.subplots(2, 4, figsize=(10, 10))

for i, dataset in enumerate([train_dataset, test_dataset]):
    for j in range(4):
        ax[i][j].imshow(dataset[j][0].numpy().transpose(2, 1, 0))
        ax[i][j].set_title(classes[dataset[j][1]])
        ax[i][j].axis("off")

criterion = nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.02
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=learning_rate
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=20,
    gamma=0.1
)


def get_batch_accuracy(outputs, labels):
    _, pred = torch.max(outputs, 1)
    correct, total = (labels == pred).sum(), len(labels)

    return 100. * float(correct) / total


def loss_acc_loader(cnn, loader):
    """
    Additional function, needed for plotting statistics
    """
    cnn.eval()
    loss, correct, total = 0., 0., 0.

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        outputs = cnn(X)

        current_loss = criterion(outputs, y)

        loss += current_loss.item()

        predicted = torch.max(outputs, 1)[1]

        correct += (predicted == y).sum().float()

        total += len(y)

    accuracy = 100 * float(correct) / total

    return accuracy, loss / total


EPOCHS = 60

epoch_data = []
train_loss_data = []
test_loss_data = []
train_acc_data = []
test_acc_data = []

net = net.to(device)
for epoch in range(EPOCHS):

    loader = tqdm(train_loader)
    for train, labels in loader:
        net.train()
        train, labels = train.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(train)

        running_loss = criterion(outputs, labels)

        loader.set_description(
            f'Loss: {round(running_loss.item() / len(labels), 6)} \
            Accuracy: {round(get_batch_accuracy(outputs, labels), 6)}'
        )

        running_loss.backward()

        optimizer.step()

    net.eval()

    tr_acc, tr_lss = loss_acc_loader(net, train_loader)
    test_acc, test_lss = loss_acc_loader(net, test_loader)

    epoch_data.append(epoch)

    train_acc_data.append(tr_acc)
    train_loss_data.append(tr_lss)

    test_acc_data.append(test_acc)
    test_loss_data.append(test_lss)

    print(
        f'Epoch: {epoch} \
        Acc train: {tr_acc} Loss train: {tr_lss} \
        Acc val {test_acc}% Loss val: {test_lss}'
    )

    scheduler.step()

f = plt.figure()

fig, ax = plt.subplots(2, 1, figsize=(8, 12))

ax[0].plot(epoch_data, train_loss_data)
ax[0].plot(epoch_data, test_loss_data)
ax[0].set_title("Model loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend(["Train", "Validation"],
             loc='upper left')

ax[0].plot()

ax[1].plot(epoch_data, train_acc_data)
ax[1].plot(epoch_data, test_acc_data)
ax[1].set_title("Model accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend(["Train", "Validation"],
             loc='upper left')

ax[1].plot()

torch.save(
    net.state_dict(),
    "/home/intern/wide-resnet18.pth.tar"
)
