import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from .model import CNN_32_32


IM_IN_SIZE = 32
IM_OUT_SIZE = 30

device = "cuda"

train_transform = transforms.Compose([
    transforms.RandomCrop(IM_OUT_SIZE),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.CenterCrop(IM_OUT_SIZE),
    transforms.ToTensor()
])

train_dataset = CIFAR100(
    root="../CIFAR100/train",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = CIFAR100(
    root="../CIFAR100/test",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

BATCH_SIZE = 512

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

net = CNN_32_32().to(device)
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

EPOCHS = 20

count = 0
loss_data = []
val_accuracy_data = []
count_data = []

for i in range(EPOCHS):
    epoch = i

    for X_batch, y_batch in train_loader:

        train = X_batch.to(device)
        labels = y_batch.to(device)

        optimizer.zero_grad()

        outputs = net(train)

        pred = torch.max(outputs, 1)[1]

        corr = (pred == labels).sum()
        accuracy_train = 100 * float(corr) / len(labels)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        count += 1

        if count % 50 == 0:

            # Calcutating test accuracy
            correct = 0
            total = 0

            for images, labels in test_loader:
                test = images.to(device)
                labels = labels.to(device)

                outputs = net(test)

                predicted = torch.max(outputs, 1)[1]

                total += len(labels)

                correct += (predicted == labels).sum()

            accuracy_val = 100 * float(correct) / total

            loss_data.append(loss.data)
            count_data.append(count)
            val_accuracy_data.append(accuracy_val)

            print(
                f'Epoch: {epoch} Iteration: {count} Loss: {loss.data}, Accuracy val: {accuracy_val}% Accuracy train: {accuracy_train}')


torch.save(
    net.state_dict(),
    '/content/drive/My Drive/Colab Notebooks/CIFAR-100/model-weights.pth'
)