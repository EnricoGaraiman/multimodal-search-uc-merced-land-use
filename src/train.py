import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import time
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import src.models as models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def training_stage(PARAMETERS, dataset_train_loader, dataset_test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('\nTraining stage on', device)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    model = models.create_model(PARAMETERS)
    model.to(device)

    print('\n---------------\nCNN Model Summary\n---------------\n')
    summary(model, input_size=(10, 3, PARAMETERS['size'], PARAMETERS['size']))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMETERS['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10, factor=0.1)

    best_epoch = 0
    best_model = ''
    count_last_best = 0
    start_time_training = time.time()
    for epoch in range(1, PARAMETERS['epochs'] + 1):
        print(
            f'\n-------------------------------------------------\nEpoch {epoch}\n-------------------------------------------------')

        train_loss, train_acc = train(model, dataset_train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = test(model, dataset_test_loader, loss_fn, device)
        scheduler.step(test_loss)

        print(
            f'\nTraining accuracy: {train_acc} % | Training loss: {train_loss} ||| Testing accuracy: {test_acc} % | Testing loss: {test_loss}')

        if len(test_acc_history) > 0 and test_acc > np.max(test_acc_history):
            best_model = 'run/best_' + str(epoch) + '.pth'
            best_epoch = epoch
            torch.save(model.state_dict(), 'run/best_' + str(epoch) + '.pth')
            count_last_best = 0
            print(f'Saved best model in epoch {epoch}')

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if epoch in [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            print(f'Training total time epoch {epoch}: {time.time() - start_time_training} seconds')
            print(f'Best in epoch: {best_epoch}')
            print(f'Best loss: {np.min(test_loss_history)}')
            print(f'Best accuracy: {np.max(test_acc_history)} %\n')

        if PARAMETERS['early_stopping_epochs_no_best'] != -1 and PARAMETERS[
            'early_stopping_epochs_no_best'] < count_last_best:
            print('Early Stopping')
            break

        count_last_best += 1

    end_time_training = time.time()
    print(f'Training total time: {end_time_training - start_time_training} seconds')
    print(f'Best in epoch: {best_epoch}')
    print(f'Best loss: {np.min(test_loss_history)}')
    print(f'Best accuracy: {np.max(test_acc_history)} %\n')

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model


def train(model, training_loader, optimizer, loss_function, device):
    total_loss = 0
    avg_acc = []

    model.train()

    for data, labels in tqdm(training_loader):
        data, labels = data.to(device), labels.to(device)

        #  classifying instances
        classifications = model(data)

        #  get accuracy
        correct_predictions = sum(torch.argmax(classifications, dim=1) == labels).item()
        avg_acc.append(correct_predictions / len(data) * 100)

        #  computing loss
        loss = loss_function(classifications, labels)
        total_loss += loss.item()

        #  zeroing optimizer gradients
        optimizer.zero_grad()

        #  computing gradients
        loss.backward()

        #  optimizing weights
        optimizer.step()

    return round(total_loss / len(training_loader), 2), round(np.array(avg_acc).mean(), 2)


def test(model, testing_loader, loss_function, device):
    total_loss = 0
    avg_acc = []

    # defining model state
    model.eval()

    with torch.no_grad():
        for data, labels in tqdm(testing_loader):
            data, labels = data.to(device), labels.to(device)

            #  classifying instances
            classifications = model(data)

            #  get accuracy
            correct_predictions = sum(torch.argmax(F.softmax(classifications, dim=1), dim=1) == labels).item()
            avg_acc.append(correct_predictions / len(data) * 100)

            #  computing loss
            loss = loss_function(classifications, labels)
            total_loss += loss.item()

    return round(total_loss / len(testing_loader), 2), round(np.array(avg_acc).mean(), 2)


def plot_training_results(train_loss_history, train_acc_history, test_loss_history, test_acc_history):
    plt.figure()
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train loss', color='brown')
    plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, label='Test loss', color='darkgreen')
    plt.title('Loss results')
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('run/loss_results.jpg', dpi=300)

    plt.figure()
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train accuracy', color='brown')
    plt.plot(range(1, len(test_acc_history) + 1), test_acc_history, label='Test accuracy', color='darkgreen')
    plt.title('Accuracy results')
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Accuracy [%]')
    ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('run/accuracy_results.jpg', dpi=300)


def plot_confusion_matrix(PARAMETERS, dataset_test_loader, best_model, classes):
    labels = range(0, PARAMETERS['num_classes'])

    model = models.create_model(PARAMETERS)

    model.load_state_dict(torch.load(best_model))
    model.eval()

    test_accuracy, predictions, real_labels = get_test_predictions(dataset_test_loader, model)

    print(f"Test dataset accuracy: {test_accuracy :>0.2f}%\n")

    cmx = confusion_matrix(real_labels, predictions, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    fig.suptitle('Confusion Matrix', fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.savefig('run/confusion-matrix.png', dpi=fig.dpi)

    return model, predictions, real_labels


def get_test_predictions(dataloader, model):
    predictions = []
    real_labels = []
    start_time_testing = []
    end_time_testing = []
    avg_acc = []

    dataloader.dataset.return_info = True

    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            start_time_testing.append(time.time())

            output = model(data)

            probabilities = F.softmax(output, dim=1)
            pred = torch.argmax(probabilities, dim=1)

            end_time_testing.append(time.time())

            correct_predictions = (pred == labels).sum().item()
            avg_acc.append(correct_predictions / len(data) * 100)

            predictions.extend(pred.numpy())
            real_labels.extend(labels.numpy())

    print(
        f'Inference time: {(np.average(end_time_testing) - np.average(start_time_testing)) / len(dataloader) * 1000} ms / image')

    return round(np.array(avg_acc).mean(), 2), predictions, real_labels
