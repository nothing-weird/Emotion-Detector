import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
# import pandas
# import natsort
# from PIL import Image
import cv2 as cv
# import io
# from skimage import io
import glob


train_root = "data/train/"
test_root = "data/test/"

batch_size = 100
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # order given by dataset

transformTrain = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()
])

transformTest = transforms.Compose([
    # ...
    transforms.ToTensor()
])


class FacialExpressionDataset(Dataset):
    """ Facial expression dataset

        - Works on windows at the time, not sure if it also works on other platforms,
          because of the slash/backslash behavior in windows
          -> need to check for a compatible code with glob.glob(...)
        - Dataset must be located in the same directory
        - How do we upload the dataset into github, it's too big
    """
    def __init__(self, root_dir, transform=None):
        """Initialisation"""
        self.root_dir = root_dir
        self.transform = transform
        fileList = glob.glob(root_dir + "*")
        print(fileList)
        self.data = []
        for emotionPath in fileList:
            emotionName = emotionPath.split("\\")[1]
            repairedEmotionPath = root_dir + "/" + emotionName
            imgPathList = glob.glob(repairedEmotionPath + "/*.jpg")
            for imgPath in imgPathList:
                self.data.append([imgPath, emotionName])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgPath, emotionName = self.data[index]
        img = cv.imread(imgPath)
        tensorImg = self.transform(img).sum(dim = 0).unsqueeze(dim=0)
        label = classes.index(emotionName)
        return tensorImg, label


if __name__ == '__main__':

    train_set = FacialExpressionDataset(train_root, transform=transformTrain)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = FacialExpressionDataset(test_root, transform=transformTest)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    print(train_set.__len__())
    sample = next(iter(train_set))
    image, emotion = sample
    # print(len(sample))
    # print(image.shape)
    image = image.sum(dim = 0)
    # plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

    batch = next(iter(train_loader))
    print(len(batch))

    images, labels = batch
    print('Images Shape: ')
    print(images.shape)
    # print(images.unsqueeze(dim=1).shape)
    # images = images.unsqueeze(dim=1)

    grid = torchvision.utils.make_grid(images, nrow=10)
    print(grid.shape)
    grid = grid.sum(dim=0).unsqueeze(dim=0)
    print(grid.shape)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)), cmap='gray')
    plt.show()
    print("labels: ", labels)