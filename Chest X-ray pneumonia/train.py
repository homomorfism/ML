import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from .model import CNN
from .dataset import Xray_dataset


device = 'cuda'
ROOT = "/content/drive/My Drive/Colab Notebooks/Chest-Xray/data"


def calc_accuracy(loader, net):
    """
    Function that calculates accuracy (for evaluating the model)
    """
    correct, total = 0., 0.
    for X, y in loader:
        test = X.to(device)
        labels = y.view(-1, 1).to(device)

        outputs = net(test)

        predicted = torch.round(torch.sigmoid(outputs))

        correct += (labels == predicted).sum().float()

        total += len(X)

    return 100 * correct / total


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = Xray_dataset(
    what='train',
    transform=transform
)
val_dataset = Xray_dataset(
    what='val',
    transform=transform
)
test_dataset = Xray_dataset(
    what='test',
    transform=transform
)

BATCH_SIZE = 16

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Displaying data samples with labels
f = plt.figure()
f, ax = plt.subplots(1, 2, figsize=(25, 25))
pneumonia, norm = None, None
for image, label in train_dataset:
    if label == 0:
        norm = image[0]
        break

for image, label in train_dataset:
    if label == 1:
        pneumonia = image[0]
        break

ax[0].imshow(norm)
ax[0].set_title("NORMAL")
ax[0].axis("off")
ax[0].plot()

ax[1].imshow(pneumonia)
ax[1].set_title("PNEUMONIA")
ax[1].axis("off")
ax[1].plot()

net = CNN()

print(
    net
)

criterion = nn.BCEWithLogitsLoss()

learning_rate = 0.005

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

EPOCHS = 10

loss_data = []
count = 0
train_accuracy_data = []
val_accuracy_data = []
count_data = []

net = net.to(device)

for epoch in range(EPOCHS):

    for X_batch, y_batch in tqdm(train_loader):
        train = X_batch.to(device)
        labels = y_batch.view(-1, 1).type(dtype=torch.float).to(device)

        optimizer.zero_grad()

        outputs = net(train)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        count += 1

        if count % 50 == 0:
            loss_data.append(loss.data)
            count_data.append(count)

            val_accuracy = calc_accuracy(val_loader, net)
            val_accuracy_data.append(val_accuracy)

            print(
                f'Epoch: {epoch} Iteration: {count} Loss: {loss.data} Accuracy val: {val_accuracy}'
            )

# Plotting statistics: Number of iterations vs. training loss and Validation accuracy
f = plt.figure()
f, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].plot(count_data, loss_data)
ax[0].set_title("Number of iterations vs. training loss")
ax[0].set_xlabel("Iterations")
ax[0].set_ylabel("Training loss")
ax[0].plot()

ax[1].plot(count_data, val_accuracy_data)
ax[1].set_title("Number of iterations vs. Validation accuracy")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Accuracy")
ax[1].plot()

# Saving the model weights
torch.save(
    net.state_dict(),
    '/content/drive/My Drive/Colab Notebooks/Chest-Xray/model-weights.pth'
)
