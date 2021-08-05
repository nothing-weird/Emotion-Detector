import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import glob
import numpy as np


train_root = "FER-2013/train/"
test_root = "FER-2013/test/"

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # order given by dataset

transformTrain = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(44),
    transforms.RandomRotation(6),
    transforms.Normalize(1.5504047870635986, 0.742726743221283, inplace=True)
])

transformTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(1.5504047870635986, 0.742726743221283, inplace=True)
])


class FacialExpressionDataset(Dataset):
    """ Facial expression dataset
        - Works on windows at the time, not sure if it also works on other platforms,
          because of the slash/backslash behavior in windows
          -> need to check for a compatible code with glob.glob(...)
        - Dataset must be located in the same directory
    """
    def __init__(self, root_dir: str, transform: transforms.Compose):
        """
        Initializes the FacialExpressionDataset from kaggle

        :param root_dir: Directory for the dataset
        :param transform: Transformation to perform on the images
        """
        self.root_dir = root_dir
        self.transform = transform
        fileList = glob.glob(root_dir + "*")
        self.data = []
        for emotionPath in fileList:
            emotionName = emotionPath.split("\\")[1]
            repairedEmotionPath = root_dir + "/" + emotionName
            imgPathList = glob.glob(repairedEmotionPath + "/*.jpg")
            for imgPath in imgPathList:
                self.data.append([imgPath, emotionName])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> (torch.Tensor, int):
        imgPath, emotionName = self.data[index]
        img = cv.imread(imgPath)
        tensorImg = self.transform(img).sum(dim = 0).unsqueeze(dim=0)
        label = classes.index(emotionName)
        return tensorImg, label


def get_normalization(dataloader: DataLoader) -> (int, int):
    """
    Calculates the mean and standard deviation for the given images of the dataset.

    :param dataloader: Dataset
    :return: mean, std
    """
    mean = []
    std = []

    for i, (images, labels) in enumerate(dataloader):
        numpy_image = images.numpy()
        batch_m = np.mean(numpy_image, axis=(0,2,3))
        batch_std = np.std(numpy_image, axis=(0,2,3))

        mean.append(batch_m)
        std.append(batch_std)

    mean = np.array(mean).mean(axis=0).item()
    std = np.array(std).mean(axis=0).item()
    return mean, std


if __name__ == '__main__':
    train_set = FacialExpressionDataset(train_root, transformTrain)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    m, s = get_normalization(train_loader)
    batch = next(iter(train_loader))
    images, labels = batch
    print(images.shape)
    print('mean:', m)  # = 1.5504047870635986
    print('Std:', s)  # = 0.742726743221283