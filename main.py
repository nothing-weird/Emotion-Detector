# Imports
import torch
import torchvision.transforms as transforms
import resnetModel
import facialExpressionDataset as faceSet
import gui_test
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sys import platform
import sys
import cv2
import numpy as np


# Constants
device = torch.device('cpu')
labels = faceSet.classes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images = []
# Add dummy_image to image history list, that way the program doesn't crash when no face is detected
# in the first frame.
dummy_image = torch.randn(1,1,44,44)
images.append(dummy_image)

# Load model
path_to_model = 'emotion_detection_resnet_model2.pth'  # best model for now (close to 70 % accuracy on test_set)
model = resnetModel.ResNet18()
model = model.to(device)
model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))


# Methods for main programm execution

# Returns the probability for each emotion in predictions
# and the predicted emotion label (int) in predictedEmotion -> for real emotion see labels constant (index over array)
def predict_emotion(net: resnetModel.ResNet18(), preprocessed_image: torch.Tensor) -> (list, int):
    """

    :param net: Model to predict emotions
    :param preprocessed_image: image from video-feed with the necessary preprocesses
    :return: Returns the probability for each emotion in predictions as a list
            and the predicted emotion label (int) in predictedEmotion
            -> for real emotion see labels constant (index over array)
    """
    predictions = net(preprocessed_image)
    predictedEmotion = predictions.argmax(dim=1).item()

    return predictions.squeeze().tolist(), predictedEmotion


def testDevice():
    '''
    This function checks if there is a camera available.
    First it checks if a camera is connected.
    Then it checks if the camera is already used by another program.
    If not available --> messagebox appears and program ends
    '''
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()):
        print("Camera conntected")
    else:
        print("Alert ! Camera disconnected")
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror(title=None, message='No camera connected')
        sys.exit()
    number_itteration = 0

    try:
        ret, frame = cap.read()
        if number_itteration == 0:
            print(len(frame))
            print('Camera is available')
    except Exception as e:
        print('Camera already used')
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror(title=None, message='Camera already used by another application')
        sys.exit()
    number_itteration += 1


def show_frames():
    resized = None
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    flip_img = cv2.flip(cv2image, 1)

    # convert the image flip_img into grayscale so it can be used for face detection
    gray = cv2.cvtColor(flip_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw the square around the face if there is a face
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(flip_img, (x, y), (x + w, y + h), (0, 255, 180), 2)
            roi_gray = gray[y:y + h, x:x + w]
            resized = cv2.resize(roi_gray, (44, 44), interpolation=cv2.INTER_AREA)

    img = Image.fromarray(flip_img)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    camera_view.imgtk = imgtk
    camera_view.configure(image=imgtk)
    return resized


def prepare_image(resized:np.ndarray) -> torch.Tensor:
    """
    Transforms image returned by show_frames().
    If no image is found through the face-cascade classifier form openCV in show_frames(),
    the last image from the history is returned.
    Prevents the programm from crashing when no face is found.

    :param resized: image from camera feed
    :return: Returns an image as torch.Tensor in the appropriate shape for the model.
    """
    if resized is None:
        return images[-1]
    else:
        transform = transforms.Compose([
            # ...
            transforms.ToTensor()
        ])
        img = transform(resized)
        img = img.unsqueeze(dim=0)
        return img


def update_all(root, obj):
    img_for_model = show_frames()
    img_for_model = prepare_image(img_for_model)
    preds, pred_emotion = predict_emotion(model, img_for_model)
    obj.update_emoji(pred_emotion)
    obj.update_boarder(max(preds))
    obj.update_percentage(preds)
    obj.update_recognized_emotion(pred_emotion)
    root.after(80, func=lambda: update_all(root, obj))
    return img_for_model


if __name__ == '__main__':
    print("let's go")

    # check if camera is available
    testDevice()

    # Splash screen displays 3 seconds before main program starts
    splash_root = tk.Tk()

    splash_root.resizable(False, False)
    splash_root.overrideredirect(True)

    # get the screen size of your computer 
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()

    # Get the window position from the top dynamically as well as position from left or right as follows
    x_cordinate = int((screen_width / 2) - (500 / 2))
    y_cordinate = int((screen_height / 2) - (500 / 2))

    splash_root.geometry("{}x{}+{}+{}".format(500, 500, x_cordinate, y_cordinate))
    splash_img = tk.PhotoImage(file='buttons/splash_screen.png')
    splash_img_label = tk.Label(image=splash_img)
    splash_img_label.pack()
    splash_root.after(3000, splash_root.destroy)
    splash_root.mainloop()

    root = tk.Tk()

    # window size configuration
    if platform == "linux" or platform == "linux2":
        root.attributes('-zoomed', True)
    elif platform == "win32":
        root.wm_state("zoomed")

    # general settings
    root.title("Emojifier")
    root.configure(background="#122E40")
    root.iconbitmap('buttons/icon.ico')
    root.minsize(1200, 600)

    camera_view = tk.Label(root, bg="#F2BB77")
    #camera_view.place(x=60, y=120)
    camera_view.place(x=-70, y=70, relx=0.1, rely=0.05)
    cap = cv2.VideoCapture(0)

    ob = gui_test.GuiWindow(root)

    last_image = update_all(root, ob)

    images.append(last_image)
    if len(images) >= 2:
        images.pop()
    print(len(images))
    # close application with Esc key
    root.bind("<Escape>", lambda x: root.destroy())

    root.mainloop()
