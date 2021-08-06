import collections
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import facialExpressionDataset as mySet
import tqdm
import glob
import simpleModel
import complexModel
import customModel
import vggModel
import resnetModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
# import plotting
import itertools

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learningRate = 0.001  # 0,01
maxLearningRate = 0.01
numEpochs = 100
batchSize = 256  # don't change... dobrenas code for plotting is depending on it
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Networks to train and test:
simple_model = simpleModel.Net()
simple_model = simple_model.to(device)
complex_model = complexModel.NeuralNet()
complex_model = complex_model.to(device)
custom_model = customModel.NeuralNet()
custom_model = custom_model.to(device)
vgg_model = vggModel.VGG_Net()
vgg_model = vgg_model.to(device)
resnet_model = resnetModel.ResNet18()
resnet_model = resnet_model.to(device)

# Train and test sets:
train_set = mySet.FacialExpressionDataset(mySet.train_root, mySet.transformTrain)
test_set = mySet.FacialExpressionDataset(mySet.test_root, mySet.transformTest)
train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=True)

# Optimizers and Schedulers for each model (they all have different parameters)
optimizer_simple_model = optim.SGD(simple_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=5e-4)
optimizer_complex_model = optim.SGD(complex_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
optimizer_custom_model = optim.SGD(custom_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
optimizer_vgg_model = optim.SGD(vgg_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
optimizer_resnet_model = optim.SGD(resnet_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=5e-4)

scheduler_simple_model = torch.optim.lr_scheduler.CyclicLR(optimizer_simple_model,
                                                           max_lr=maxLearningRate, base_lr=learningRate)
scheduler_complex_model = torch.optim.lr_scheduler.CyclicLR(optimizer_complex_model,
                                                            max_lr=maxLearningRate, base_lr=learningRate)
scheduler_custom_model = torch.optim.lr_scheduler.CyclicLR(optimizer_custom_model,
                                                           max_lr=maxLearningRate, base_lr=learningRate)
scheduler_vgg_model = torch.optim.lr_scheduler.CyclicLR(optimizer_vgg_model,
                                                        max_lr=maxLearningRate, base_lr=learningRate)
scheduler_resnet_model = torch.optim.lr_scheduler.CyclicLR(optimizer_resnet_model,
                                                           max_lr=maxLearningRate, base_lr=learningRate)


# Methods for training
# ====================


def get_num_correct(preds: torch.Tensor, labels: torch.Tensor) -> int:
    """
    Compare predicted emotions with the actual emotions.

    :param preds: torch tensor of shape (1, 7) containing the probabilities for one prediction of an image.
    :param labels: integer value for the actual emotion.
    :return: Sum of the correctly predicted emotions as an integer
    """
    predictions = preds.argmax(dim=1, keepdim=True)
    return predictions.eq(labels.view_as(predictions)).sum().item()


# Retruns current learning rate
def get_lr(optimizer: optim.SGD) -> float:
    """

    :param optimizer: Used torch.optim optimizer
    :return: Current learning rate. Used to display while training.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model: resnetModel.ResNet18(),
          train_data_loader: DataLoader,
          optimizer: optim.SGD,
          scheduler: optim.lr_scheduler) -> (list, list):
    """
    Training of the specified model with all images of the dataset, by iterating over all batches of
    the data-loader for the train set.
    The nn.CrossEntropyLoss is used as the loss function

    :param model:
    :param train_data_loader:
    :param optimizer:
    :param scheduler:
    :return: train_loss, train_acc: lists with the loss and accuracy (in %) for each batch
    """
    model.train()
    pbar = tqdm.tqdm(train_data_loader)
    running_loss = 0.0
    correct = 0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    train_acc_epoch = []
    train_loss_epoch = []

    # Training loop for batchsize:
    for index_batch, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += get_num_correct(prediction, labels)
        processed += len(images)
        accuracy = 100. * correct / processed
        train_acc.append(accuracy)
        train_loss.append(loss.item())
        if index_batch % 10 == 0:
            print(f'Loss={loss.item()} Batch_id={index_batch} LR={get_lr(optimizer)} '
                  f'Accuracy={accuracy:0.2f} %')
        if (index_batch % 112 == 0) and (index_batch != 0):
            train_acc_epoch.append(accuracy)
            train_loss_epoch.append(loss.item())
            print('Current train_acc_epoch: ' + str(train_acc_epoch))
            print('Current train_loss_epoch: ' + str(train_loss_epoch))

    return train_loss, train_acc, train_acc_epoch, train_loss_epoch


def test(model: resnetModel.ResNet18(),
         test_data_loader:  DataLoader) -> (list, list):
    """
    Testing of the specified model with all images, by iterating over all batches of
    the data-loader for the test set.

    :param model:
    :param test_data_loader:
    :return: test_loss, test_acc: Values for each epoch
    """
    model.eval()
    testing_loss = 0
    running_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = []
    test_acc = []

    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            loss = criterion(prediction, labels).item()
            running_loss += loss
            correct += get_num_correct(prediction, labels)

    # testing_loss = testing_loss / len(labels)
    test_loss.append(loss)

    accuracy = 100 * correct / len(test_data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss[-1], correct,
        len(test_data_loader.dataset),
        accuracy
    ))
    test_acc.append(accuracy)

    return test_loss, test_acc


def train_test_with_epochs(model: resnetModel.ResNet18(),
                           train_load: DataLoader,
                           test_load: DataLoader,
                           optimizer: optim.SGD,
                           scheduler: optim.lr_scheduler,
                           Epochs: int) -> list:
    """
    Global method to train and test the specified model by iterating over the number of epochs.
    For each epoch a checkpoint with the model_state_dict is saved and the loss and accuracy data is tracked.

    :param model: Model to train and test
    :param train_load: Dataloader for train set
    :param test_load: Dataloader for test set
    :param optimizer: torch.optim optimizer
    :param scheduler: Learning rate scheduler
    :param Epochs: Number of Epochs
    :return: [global_train_loss, global_train_acc, global_test_loss, global_test_acc,
                global_train_acc_epoch, global_train_loss_epoch]:
             List containing the lists for all train and test loss and accuracy data.
    """
    global_train_loss, global_train_acc = [], []
    global_test_loss, global_test_acc = [], []
    global_train_acc_epoch, global_train_loss_epoch = [], []
    print('Training of the model: ', repr(model))

    for epoch in range(Epochs):
        print("EPOCH: %s LR: %s " % (epoch, get_lr(optimizer)))
        train_loss, train_acc, train_acc_epoch, train_loss_epoch = train(model, train_load, optimizer, scheduler)
        global_train_loss.extend(train_loss), global_train_acc.extend(train_acc)
        global_train_acc_epoch.extend(train_acc_epoch), global_train_loss_epoch.extend(train_loss_epoch)
        test_loss, test_acc = test(model, test_load)
        global_test_loss.extend(test_loss), global_test_acc.extend(test_acc)

        save_checkpoint_to_folder(model, epoch, optimizer, train_acc[-1], test_acc[-1])

    print('Training completed! \n\n')

    return [global_train_loss, global_train_acc, global_test_loss, global_test_acc,
            global_train_acc_epoch, global_train_loss_epoch]


def save_checkpoint_to_folder(model: resnetModel.ResNet18(),
                              epoch: int,
                              optimizer: optim.SGD,
                              train_acc: float,
                              test_acc: float):

    """
    Saves a checkpoint with the epoch, model_state_dict, optim_state_dict, train and test accuracy to the
    Checkpoints/ folder as e.g. state_dict_epoch1.pth for epoch 1.

    :param model:
    :param epoch:
    :param optimizer:
    :param train_acc:
    :param test_acc:
    """
    path_to_checkpoint = 'Checkpoints/state_dict_epoch' + str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc
    }, path_to_checkpoint)


def find_best_checkpoint(root_dir: str = 'Checkpoints/') -> dict[int, collections.OrderedDict, dict, float, float]:
    """
    Iterates through all checkpoints in the specified folder and searches for the one with the highest accuracy.

    :param root_dir: Folder for the Checkpoints
    :return: Checkpoint with highest test accuracy. Structure of checkpoint: 'epoch': int, 'model_state_dict': dict,
             'optimizer_state_dict': dict, 'train_acc': float, 'test_acc': float
    """
    file_list_checkpoints = glob.glob(root_dir + '*')
    temp_acc = 0.0
    best_checkpoint = torch.load(file_list_checkpoints[0])

    for path in file_list_checkpoints:
        checkpoint = torch.load(path)
        acc = checkpoint['train_acc']
        if acc > temp_acc:
            temp_acc = acc
            best_checkpoint = checkpoint

    return best_checkpoint


# just some extraced testing code from main
def resnet_test_load_checkpoint():
    networkLoad = resnetModel.ResNet18()
    networkLoad.to(device)
    checkpoint = torch.load('Checkpoints/state_dict_epoch36.pth')
    print(checkpoint['train_acc'])
    networkLoad.load_state_dict(checkpoint['model_state_dict'])
    batch = next(iter(train_loader))
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    pred = networkLoad(images)
    acc = get_num_correct(pred, labels) / len(labels)
    print('Accuracy for loaded checkpoint:', acc)


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    for i, j in itertools.product(range(df_confusion.shape[0]), range(df_confusion.shape[1])):
        plt.text(i, j, format(df_confusion.iloc[i, j], fmt), horizontalalignment="center",
                 color="black")

    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def plotting():
    results_resnet_model = train_test_with_epochs(resnet_model, train_loader, test_loader,
                                                  optimizer_resnet_model, scheduler_resnet_model, numEpochs)

    # Defining some variables for easy access
    train_acc = results_resnet_model[1]
    test_acc = results_resnet_model[3]
    train_loss = results_resnet_model[0]
    test_loss = results_resnet_model[2]
    train_acc_epoch = results_resnet_model[4]
    train_loss_epoch = results_resnet_model[5]

    #---------------------------------------------------------
    # Accuracy: Plotting for train shows accuracy for every batch,
    #           plotting for test shows accuracy after every epoch.
    plt.plot([x / 113 for x in range(1, len(train_acc) + 1)], train_acc, color='cyan')
    plt.plot([x for x in range(1, len(train_acc_epoch) + 1)], train_acc_epoch, 'x', color='blue')
    plt.plot([x for x in range(len(test_acc))], test_acc, color='orange')

    # Title, labels, legend markers
    plt.title('Accuracy Data for LR = ' + str(learningRate) + ', Model is ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(0, 0, color='cyan', Label='Training accuracy for all batches in every epoch')
    plt.plot(0, 0, color='orange', Label='Testing accuracy for every epoch')
    plt.plot(1, train_acc_epoch[0], 'x', color='blue', Label='Training accuracy at the end of each epoch')
    plt.legend()
    plt.show()

    #---------------------------------------------------------
    # Loss: Plotting for train shows accuracy for every batch,
    #       plotting for test shows accuracy after every epoch.
    plt.plot([x / 113 for x in range(1, len(train_loss) + 1)], train_loss, color='cyan')
    plt.plot([x for x in range(1, len(train_loss_epoch) + 1)], train_loss_epoch, 'x', color='blue')
    plt.plot([x for x in range(len(test_loss))], test_loss, color='orange')

    # Title, labels, legend markers
    plt.title('Loss Data for LR = ' + str(learningRate) + ', Model is ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(0, 0, color='cyan', Label='Training loss for all batches in every epoch')
    plt.plot(0, 0, color='orange', Label='Testing loss for every epoch')
    plt.plot(1, train_loss_epoch[0], 'x', color='blue', Label='Training loss at the end of each epoch')
    plt.legend()
    plt.show()

    #---------------------------------------------------------
    # Loss plot per epoch, NOT per batch
    plt.plot([x for x in range(1, len(train_loss_epoch) + 1)], train_loss_epoch, color='blue')
    plt.title('Loss Data per Epoch, LR = ' + str(learningRate) + 'Model ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_data(data1: list,
              legend1: str,
              x_label: str,
              y_label: str,
              title: str,
              result_path: str = '',
              data2: list = None,
              legend2: str = None) -> None:

    axis_font = {'fontname': 'Arial', 'size': '16'}
    title_font = {'fontname': 'Arial', 'size': '20'}

    filename = title+"_"+datetime.now().strftime("%d-%m-%Y_%H-%M")

    plt.plot([int(x) for x in range(len(data1))], data1, color = 'cyan', label = legend1)
    if data2 is not None:
        plt.plot([int(x) for x in range(len(data2))], data2, color = 'orange', label = legend2)

    plt.xlabel(x_label, **axis_font)
    plt.ylabel(y_label, **axis_font)
    plt.title(title, **title_font)
    plt.legend()

    outname = os.path.join(result_path, filename + '.png')
    plt.savefig(outname)
    plt.show()
    plt.cla()
    plt.close()
    print("Data plotted and saved as: {}".format(outname))


if __name__ == '__main__':

    # checking for cuda
    print(device)

    # training
    resnet_t_loss, resnet_t_acc, resnet_test_loss, resnet_test_acc, resnet_t_acc_global, resnet_t_loss_global\
        = train_test_with_epochs(resnet_model, train_loader, test_loader,
                                 optimizer_resnet_model, scheduler_resnet_model, numEpochs)

    # plot acc for epochs
    plot_data(data1 = resnet_t_acc_global,
              legend1= 'train data',
              data2 = resnet_test_acc,
              legend2= 'test data',
              x_label = 'Epochs',
              y_label = 'Accuracy in %',
              title = 'Comparison of test and train accuracy for ResNet')

    # plot loss for epochs
    plot_data(data1 = resnet_t_loss_global,
              legend1= 'train loss',
              data2 = resnet_test_loss,
              legend2= 'test loss',
              x_label = 'epochs',
              y_label = 'cross entropy loss',
              title = 'Comparison of test and train loss for ResNet')

    # plot acc for training in batches
    plot_data(data1 = resnet_t_acc,
              legend1 = 'train accuracy',
              x_label = 'batch (256 images per batch)',
              y_label = 'accuracy in %',
              title = 'Training of ResNet')

    ####### Path for model #######
    path_resnet = 'emotion_detection_resnet_model7.pth'
    path_vgg = 'emotion_detection_vgg_model.pth'
    path_custom = 'emotion_detection_custom_model.pth'
    path_complex = 'emotion_detection_complex_model.pth'

    ####### Find best checkpoint in folder #######
    checkpoint_with_highest_acc = find_best_checkpoint()
    print('Epoch: ', checkpoint_with_highest_acc['epoch'])
    print('Train accuracy: ', checkpoint_with_highest_acc['train_acc'])
    print('Test accuracy: ', checkpoint_with_highest_acc['test_acc'])

    ### Save model state ###
    torch.save(checkpoint_with_highest_acc['model_state_dict'], path_resnet)

    ### Confusion matrix ###
    networkLoad = resnetModel.ResNet18()
    networkLoad.to(device)
    checkpoint_with_highest_acc = torch.load('emotion_detection_resnet_model2.pth')
    networkLoad.load_state_dict(checkpoint_with_highest_acc)

    t_loss, t_acc = test(networkLoad, test_loader)
    print('Loss for resnet2: ', t_loss[-1])
    print('Acc for resnet2: ', t_acc[-1])

    networkLoad.eval()

    predictions = []
    groundtruth = []

    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data, target
            images, labels = images.to(device), labels.to(device)

            outputs = networkLoad(images)
            predictions.append(outputs.max(1, keepdim=True)[1])  # indices of max values
            groundtruth.append(labels)

    for idx, prediction in enumerate(predictions):
        predictions[idx] = prediction.cpu().numpy()
        groundtruth[idx] = groundtruth[idx].cpu().numpy()
    predictions = np.concatenate(predictions)
    groundtruth = np.concatenate(groundtruth)

    groundtruth = pd.Series(groundtruth, name='True label')
    predictions = pd.Series(predictions.reshape(predictions.shape[0]), name='Predicted label')
    df_confusion_matrix = pd.crosstab(groundtruth, predictions)
    df_confusion_matrix_percent = pd.crosstab(groundtruth, predictions, normalize='index').round(6) * 100

    print(df_confusion_matrix)
    print(df_confusion_matrix_percent)
    print(type(df_confusion_matrix))
    print(df_confusion_matrix[0])
    plot_confusion_matrix(df_confusion=df_confusion_matrix_percent)