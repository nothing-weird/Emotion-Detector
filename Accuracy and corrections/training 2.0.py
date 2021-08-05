
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import facialExpressionDataset as mySet
import matplotlib.pyplot as plt
import tqdm
import simpleModel
import complexModel
import customModel

# Hyperparameters
device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')
kernel_size = (5, 5)
numEmotions = 7
learningRate = 0.0001  # 0,05
maxLearningRate = 0.1
numEpochs = 40
batchSize = 128

# Networks to train and test:
simple_model = simpleModel.Net()
simple_model = simple_model.to(device)
complex_model = complexModel.NeuralNet()
complex_model = complex_model.to(device)
custom_model = customModel.NeuralNet()
custom_model = custom_model.to(device)

# Train and test sets:
train_set = mySet.FacialExpressionDataset(mySet.train_root, mySet.transformTrain)
test_set = mySet.FacialExpressionDataset(mySet.test_root, mySet.transformTest)
train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=True)

# Optimizer and Scheduler
optimizer_simple_model = optim.SGD(simple_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
optimizer_complex_model = optim.SGD(complex_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
optimizer_custom_model = optim.SGD(custom_model.parameters(), lr=learningRate, momentum=0.9, weight_decay=9e-4)
scheduler_simple_model = torch.optim.lr_scheduler.CyclicLR(optimizer_simple_model,
                                                           max_lr=maxLearningRate, base_lr=learningRate)
scheduler_complex_model = torch.optim.lr_scheduler.CyclicLR(optimizer_complex_model,
                                                            max_lr=maxLearningRate, base_lr=learningRate)
scheduler_custom_model = torch.optim.lr_scheduler.CyclicLR(optimizer_custom_model,
                                                           max_lr=maxLearningRate, base_lr=learningRate)


# Methods for training
# ====================

# Compare predicted emotions with the actual emotions.
# Returns the sum of correctly predicted emotions.
def get_num_correct(preds, labels):
    predictions = preds.argmax(dim=1, keepdim=True)
    return predictions.eq(labels.view_as(predictions)).sum().item()


# Retruns current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_data_loader, optimizer, scheduler):
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
        if (index_batch % 224 == 0) and (index_batch != 0):
            train_acc_epoch.append(accuracy)
            train_loss_epoch.append(loss.item())
            print('Current train_acc_epoch: ' + str(train_acc_epoch))
            print('Current train_loss_epoch: ' + str(train_loss_epoch))

    return train_loss, train_acc, train_acc_epoch, train_loss_epoch


def test(model, test_data_loader):
    model.eval()
    testing_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = []
    test_acc = []

    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            testing_loss += criterion(prediction, labels).item()
            correct += get_num_correct(prediction, labels)

    testing_loss = testing_loss / len(test_data_loader.dataset)
    test_loss.append(testing_loss)

    accuracy = 100 * correct / len(test_data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        testing_loss, correct,
        len(test_data_loader.dataset),
        accuracy
    ))
    test_acc.append(accuracy)

    return test_loss, test_acc


def train_with_epochs(model, train_load, test_load, optimizer, scheduler, Epochs):
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
        save_checkpoint_to_folder(model, epoch, optimizer, train_loss[-1], test_loss[-1])

    print('Training completed! \n\n')
    return [global_train_loss, global_train_acc, global_test_loss, global_test_acc,
            global_train_acc_epoch, global_train_loss_epoch]


def save_checkpoint_to_folder(model, epoch, optimizer, train_loss, test_loss):
    path_to_checkpoint = 'Checkpoints/state_dict_epoch' + str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }, path_to_checkpoint)


def plotting():

    train_acc = results_complex_model[1]
    test_acc = results_complex_model[3]
    train_loss = results_complex_model[0]
    test_loss = results_complex_model[2]
    train_acc_epoch = results_complex_model[4]
    train_loss_epoch = results_complex_model[5]

    # Accuracy: Plotting for train shows accuracy for every batch,
    #           plotting for test shows accuracy after every epoch.

    plt.plot([x / 225 for x in range(1, len(train_acc) + 1)], train_acc, color='cyan')
    plt.plot([x for x in range(1, len(train_acc_epoch) + 1)], train_acc_epoch, 'x', color='blue')
    plt.plot([x for x in range(len(test_acc))], test_acc, color='orange')

    plt.title('Accuracy Data for LR = ' + str(learningRate))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(0, 0, color='cyan', Label='Training accuracy for all batches in every epoch')
    plt.plot(0, 0, color='orange', Label='Testing accuracy for every epoch')
    plt.plot(1, train_acc_epoch[0], 'x', color='blue', Label='Training accuracy at the end of each epoch')
    plt.legend()
    plt.show()

    # Loss: Plotting for train shows accuracy for every batch,
    #       plotting for test shows accuracy after every epoch.

    plt.plot([x / 225 for x in range(1, len(train_loss) + 1)], train_loss, color='cyan')
    plt.plot([x for x in range(1, len(train_loss_epoch) + 1)], train_loss_epoch, 'x', color='blue')
    plt.plot([x for x in range(len(test_loss))], test_loss, color='orange')

    plt.title('Loss Data for LR = ' + str(learningRate))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(0, 0, color='cyan', Label='Training loss for all batches in every epoch')
    plt.plot(0, 0, color='orange', Label='Testing loss for every epoch')
    plt.plot(1, train_loss_epoch[0], 'x', color='blue', Label='Training loss at the end of each epoch')
    plt.legend()
    plt.show()


    plt.plot([x for x in range(len(train_loss))], train_loss, color = 'blue')
    plt.title('Loss Data for LR = ' + str(learningRate))
    plt.xlabel('Batch')
    plt.ylabel('Loss')


if __name__ == '__main__':

    results_complex_model = train_with_epochs(complex_model, train_loader, test_loader,
                                              optimizer_complex_model, scheduler_complex_model, numEpochs)
    # results_custom_model = train_with_epochs(custom_model, train_loader, test_loader,
    #                                         optimizer_custom_model, scheduler_custom_model, numEpochs)

    print('Accuracy for testset: \n', results_complex_model[3])

    # print('Accuracy for trainset: \n', results_custom_model[1])
    # results_simple_model = train_with_epochs(simple_model, train_loader, test_loader,
    #                                         optimizer_simple_model, scheduler_simple_model, numEpochs)


    # print('Accuracy for testset: \n', results_simple_model[3])

    ####### Save model #######
    path_complex = 'emotion_detection_complex_model.pth'
    torch.save(complex_model.state_dict(), path_complex)
    path_custom = 'emotion_detection_custom_model.pth'

    ####### Load model and test again #######
    # networkLoad = complexModel.NeuralNet()
    # networkLoad.to(device)
    # networkLoad.load_state_dict(torch.load(path_to_save))
    # print(repr(networkLoad))
    # print(networkLoad.state_dict())