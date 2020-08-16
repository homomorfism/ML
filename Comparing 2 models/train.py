import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.notebook import tqdm

from .finetuning_parameters import device, ROOT, classes


def loss_acc_loader(cnn, loader, criterion):
    """
    Additional function, needed for plotting statistics
    """
    cnn.eval()
    loss, correct, total = 0., 0., 0.

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        outputs = cnn(X)

        current_loss = criterion(outputs, y)

        loss += current_loss.item()

        predicted = torch.max(outputs, 1)[1]

        correct += (predicted == y).sum().float()

        total += len(y)

    accuracy = 100 * float(correct) / total

    return accuracy, loss / total


def plot_confusion_matrix(y_labels, y_pred, classes):
    cm = confusion_matrix(y_labels, y_pred)

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True)
    plt.title('Model confusion matrix \nAccuracy:{0:.3f}'.format(accuracy_score(y_labels, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_loss_acc_graphs(train_loss, train_accuracy, val_loss, val_accuracy, epoch_data):
    """
    Function plots accuracy/loss graphs after training
    """
    f = plt.figure()

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    ax[0].plot(epoch_data, train_loss)
    ax[0].plot(epoch_data, val_loss)
    ax[0].set_title("Model loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(["Train", "Validation"],
                 loc='upper left')

    ax[0].plot()

    ax[1].plot(epoch_data, train_accuracy)
    ax[1].plot(epoch_data, val_accuracy)
    ax[1].set_title("Model accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(["Train", "Validation"],
                 loc='upper left')

    ax[1].plot()


def show_metrics(net, loader):
    net = net.to(device)
    net.eval()
    y_pred, y_score, y_labels, y_onehot = [], [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        y_b_onehot = y.cpu().numpy()
        y_b_onehot = (np.arange(10) == y_b_onehot[:, None]).astype(np.int32)
        y_onehot += y_b_onehot.tolist()

        pred = net(X)

        y_score += torch.softmax(pred, dim=1).tolist()

        pred = torch.max(pred, 1)[1].tolist()

        y_pred += pred

        y_labels += y.tolist()

    plot_confusion_matrix(y_labels, y_pred, classes)
    print(
        "Displaying statictics:",
        "\nAccuracy:", accuracy_score(y_labels, y_pred),
        "\nPresicion:", precision_score(y_labels, y_pred, average='weighted'),
        "\nRecall:", recall_score(y_labels, y_pred, average='weighted'),
        "\nF1:", f1_score(y_labels, y_pred, average='weighted'),
        "\nROC-AUC", roc_auc_score(y_onehot, y_score, average='weighted', multi_class='ovr')
    )


def train(net, train_loader, val_loader, test_loader):
    """
    Actual function that trains and saves the model
    """
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.RMSprop(
        params=net.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.25,
        patience=2,
        verbose=True,
        threshold=0.0001,
        cooldown=0,
        min_lr=0.00001
    )

    # EPOCHS = 40
    EPOCHS = 20
    net = net.to(device)

    epoch_data = []

    train_accuracy = []
    train_loss = []

    val_accuracy = []
    val_loss = []

    print(f"Started training {net.get_name()}")
    for epoch in range(EPOCHS):

        for X, y in tqdm(train_loader):
            net.train()

            train = X.to(device)
            labels = y.to(device)

            optimizer.zero_grad()

            outputs = net(train)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        net.eval()

        tr_acc, tr_lss = loss_acc_loader(net, train_loader, criterion)
        val_acc, val_lss = loss_acc_loader(net, val_loader, criterion)

        epoch_data.append(epoch)
        train_accuracy.append(tr_acc)
        train_loss.append(tr_lss)
        val_accuracy.append(val_acc)
        val_loss.append(val_lss)

        print(
            f'Epoch: {epoch} \
            Acc train: {tr_acc} Loss train: {tr_lss} \
            Acc val {val_acc} Loss val: {val_lss}'
        )
        scheduler.step(val_lss)

    print(f"Displaying data about training model {net.get_name()}")
    plot_loss_acc_graphs(train_loss, train_accuracy, val_loss, val_accuracy, epoch_data)

    print(f"Displaying metrics for {net.get_name()} on val data")
    show_metrics(net, val_loader)

    print(f"Saving model {net.get_name()}")
    torch.save(
        net.state_dict(),
        os.path.join(ROOT, f"{net.get_name()}_weghts.pth")
    )

    print(f"Testing {net.get_name()}")

    show_metrics(net, test_loader)
