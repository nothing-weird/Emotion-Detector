import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import random


class GuiWindow:
    '''
    This class contains all widget elements for the tkinter gui.
    The __inti__ functions has all labels and imported images.
    functions for updating the gui are beneath.
    '''

    def __init__(self, master):
        myFrame = tk.Frame(master)
        myFrame.pack()

# all initial labels listed and placed here-----------------------------------------------------------------------------
        self.header1 = tk.Label(
            master,
            text="Welcome to the ",
            fg="white",
            bg="#122E40",
        )
        #self.header1.place(x=210, y=40)
        self.header1.place(x=60, y=0, relx=0.1, rely=0.05)
        self.header2 = tk.Label(
            master,
            text="Emojifier",
            fg="#F2BB77",
            bg="#122E40",
        )
        #self.header2.place(x=450, y=40)
        self.header2.place(x=290, y=0, relx=0.1, rely=0.05)

        self.emoji_view_green = tk.Canvas(
            master,
            width=350,
            height=350,
            bg="#61808C",
            highlightbackground="#0EF700",
            highlightthickness=2
        )

        self.emoji_view_red = tk.Canvas(
            master,
            width=350,
            height=350,
            bg="#61808C",
            highlightbackground="#F7090A",
            highlightthickness=2
        )

    # List of emotions (percentage is updated by 'update_percentage' function)
        self.anger = tk.Label(
            master,
            text="Anger:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.anger.place(x=60, y=640)
        self.anger.place(x=-40, y=560, relx=0.1, rely=0.08)

        self.disgust = tk.Label(
            master,
            text="Disgust:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.disgust.place(x=60, y=665)
        self.disgust.place(x=-40, y=585, relx=0.1, rely=0.08)

        self.fear = tk.Label(
            master,
            text="Fear:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.fear.place(x=60, y=690)
        self.fear.place(x=-40, y=610, relx=0.1, rely=0.08)

        self.happiness = tk.Label(
            master,
            text="Happiness:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.happiness.place(x=60, y=715)
        self.happiness.place(x=-40, y=635, relx=0.1, rely=0.08)

        self.neutral = tk.Label(
            master,
            text="Neutral:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.neutral.place(x=250, y=640)
        self.neutral.place(x=150, y=560, relx=0.1, rely=0.08)

        self.sadness = tk.Label(
            master,
            text="Sadness:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.sadness.place(x=250, y=665)
        self.sadness.place(x=150, y=585, relx=0.1, rely=0.08)

        self.surprise = tk.Label(
            master,
            text="Surprise:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.surprise.place(x=250, y=690)
        self.surprise.place(x=150, y=610, relx=0.1, rely=0.08)

        self.detected_emotion = tk.Label(
            master,
            text="Detected Emotion:",
            fg="white",
            bg="#122E40",
            font=("Helvetica", 12)
        )
        #self.detected_emotion.place(x=940, y=550)
        self.detected_emotion.place(x=645, y=510, relx=0.2, rely=0.05)

        self.choose_emoji = tk.Label(
            master,
            text="Choose your emoji",
            fg="white",
            bg="#122E40",
        )
        #self.choose_emoji.place(x=1305, y=145)
        self.choose_emoji.place(x=845, y=115, relx=0.3, rely=0.05)

    # Font styling
        self.header1.config(font=("Futura", 25))
        self.header2.config(font=("Futura", 25))

# import of all emojis images as labels---------------------------------------------------------------------------------

    # yellow emojis
        self.emoji_happy_yellow = ImageTk.PhotoImage(Image.open("emojis/Happiness/happy_yellow.png"))
        self.emoji_happy_yellow_label = tk.Label(image=self.emoji_happy_yellow, bg="#61808C")
        self.emoji_angry_yellow = ImageTk.PhotoImage(Image.open("emojis/Anger/angry_yellow.png"))
        self.emoji_angry_yellow_label = tk.Label(image=self.emoji_angry_yellow, bg="#61808C")
        self.emoji_disgust_yellow = ImageTk.PhotoImage(Image.open("emojis/Disgust/disgust_yellow.png"))
        self.emoji_disgust_yellow_label = tk.Label(image=self.emoji_disgust_yellow, bg="#61808C")
        self.emoji_fear_yellow = ImageTk.PhotoImage(Image.open("emojis/Fear/fear_yellow.png"))
        self.emoji_fear_yellow_label = tk.Label(image=self.emoji_fear_yellow, bg="#61808C")
        self.emoji_neutral_yellow = ImageTk.PhotoImage(Image.open("emojis/Neutral/neutral_yellow.png"))
        self.emoji_neutral_yellow_label = tk.Label(image=self.emoji_neutral_yellow, bg="#61808C")
        self.emoji_sad_yellow = ImageTk.PhotoImage(Image.open("emojis/Sadness/sad_yellow.png"))
        self.emoji_sad_yellow_label = tk.Label(image=self.emoji_sad_yellow, bg="#61808C")
        self.emoji_surprise_yellow = ImageTk.PhotoImage(Image.open("emojis/Surprise/surprise_yellow.png"))
        self.emoji_surprise_yellow_label = tk.Label(image=self.emoji_surprise_yellow, bg="#61808C")
    # array with yellow emojis
        self.yellow_ar = [self.emoji_angry_yellow_label, self.emoji_disgust_yellow_label, self.emoji_fear_yellow_label, self.emoji_happy_yellow_label, self.emoji_neutral_yellow_label, self.emoji_sad_yellow_label, self.emoji_surprise_yellow_label]

    # basic emojis
        self.emoji_happy_basic = ImageTk.PhotoImage(Image.open("emojis/Happiness/happy_basic.png"))
        self.emoji_happy_basic_label = tk.Label(image=self.emoji_happy_basic, bg="#61808C")
        self.emoji_angry_basic = ImageTk.PhotoImage(Image.open("emojis/Anger/angry_basic.png"))
        self.emoji_angry_basic_label = tk.Label(image=self.emoji_angry_basic, bg="#61808C")
        self.emoji_disgust_basic = ImageTk.PhotoImage(Image.open("emojis/Disgust/disgust_basic.png"))
        self.emoji_disgust_basic_label = tk.Label(image=self.emoji_disgust_basic, bg="#61808C")
        self.emoji_fear_basic = ImageTk.PhotoImage(Image.open("emojis/Fear/fear_basic.png"))
        self.emoji_fear_basic_label = tk.Label(image=self.emoji_fear_basic, bg="#61808C")
        self.emoji_neutral_basic = ImageTk.PhotoImage(Image.open("emojis/Neutral/neutral_basic.png"))
        self.emoji_neutral_basic_label = tk.Label(image=self.emoji_neutral_basic, bg="#61808C")
        self.emoji_sad_basic = ImageTk.PhotoImage(Image.open("emojis/Sadness/sad_basic.png"))
        self.emoji_sad_basic_label = tk.Label(image=self.emoji_sad_basic, bg="#61808C")
        self.emoji_surprise_basic = ImageTk.PhotoImage(Image.open("emojis/Surprise/surprise_basic.png"))
        self.emoji_surprise_basic_label = tk.Label(image=self.emoji_surprise_basic, bg="#61808C")
    # array with basic emojis
        self.basic_ar = [self.emoji_angry_basic_label, self.emoji_disgust_basic_label, self.emoji_fear_basic_label, self.emoji_happy_basic_label, self.emoji_neutral_basic_label, self.emoji_sad_basic_label, self.emoji_surprise_basic_label]

    # boy emojis
        self.emoji_happy_boy = ImageTk.PhotoImage(Image.open("emojis/Happiness/happy_boy.png"))
        self.emoji_happy_boy_label = tk.Label(image=self.emoji_happy_boy, bg="#61808C")
        self.emoji_angry_boy = ImageTk.PhotoImage(Image.open("emojis/Anger/angry_boy.png"))
        self.emoji_angry_boy_label = tk.Label(image=self.emoji_angry_boy, bg="#61808C")
        self.emoji_disgust_boy = ImageTk.PhotoImage(Image.open("emojis/Disgust/disgust_boy.png"))
        self.emoji_disgust_boy_label = tk.Label(image=self.emoji_disgust_boy, bg="#61808C")
        self.emoji_fear_boy = ImageTk.PhotoImage(Image.open("emojis/Fear/fear_boy.png"))
        self.emoji_fear_boy_label = tk.Label(image=self.emoji_fear_boy, bg="#61808C")
        self.emoji_neutral_boy = ImageTk.PhotoImage(Image.open("emojis/Neutral/neutral_boy.png"))
        self.emoji_neutral_boy_label = tk.Label(image=self.emoji_neutral_boy, bg="#61808C")
        self.emoji_sad_boy = ImageTk.PhotoImage(Image.open("emojis/Sadness/sad_boy.png"))
        self.emoji_sad_boy_label = tk.Label(image=self.emoji_sad_boy, bg="#61808C")
        self.emoji_surprise_boy = ImageTk.PhotoImage(Image.open("emojis/Surprise/surprise_boy.png"))
        self.emoji_surprise_boy_label = tk.Label(image=self.emoji_surprise_boy, bg="#61808C")
    # array with boy emojis
        self.boy_ar = [self.emoji_angry_boy_label, self.emoji_disgust_boy_label, self.emoji_fear_boy_label, self.emoji_happy_boy_label, self.emoji_neutral_boy_label, self.emoji_sad_boy_label, self.emoji_surprise_boy_label]

    # halloween emojis
        self.emoji_happy_halloween = ImageTk.PhotoImage(Image.open("emojis/Happiness/happy_halloween.png"))
        self.emoji_happy_halloween_label = tk.Label(image=self.emoji_happy_halloween, bg="#61808C")
        self.emoji_angry_halloween = ImageTk.PhotoImage(Image.open("emojis/Anger/angry_halloween.png"))
        self.emoji_angry_halloween_label = tk.Label(image=self.emoji_angry_halloween, bg="#61808C")
        self.emoji_disgust_halloween = ImageTk.PhotoImage(Image.open("emojis/Disgust/disgust_halloween.png"))
        self.emoji_disgust_halloween_label = tk.Label(image=self.emoji_disgust_halloween, bg="#61808C")
        self.emoji_fear_halloween = ImageTk.PhotoImage(Image.open("emojis/Fear/fear_halloween.png"))
        self.emoji_fear_halloween_label = tk.Label(image=self.emoji_fear_halloween, bg="#61808C")
        self.emoji_neutral_halloween = ImageTk.PhotoImage(Image.open("emojis/Neutral/neutral_halloween.png"))
        self.emoji_neutral_halloween_label = tk.Label(image=self.emoji_neutral_halloween, bg="#61808C")
        self.emoji_sad_halloween = ImageTk.PhotoImage(Image.open("emojis/Sadness/sad_halloween.png"))
        self.emoji_sad_halloween_label = tk.Label(image=self.emoji_sad_halloween, bg="#61808C")
        self.emoji_surprise_halloween = ImageTk.PhotoImage(Image.open("emojis/Surprise/surprise_halloween.png"))
        self.emoji_surprise_halloween_label = tk.Label(image=self.emoji_surprise_halloween, bg="#61808C")
    # array with halloween emojis
        self.halloween_ar = [self.emoji_angry_halloween_label, self.emoji_disgust_halloween_label, self.emoji_fear_halloween_label, self.emoji_happy_halloween_label, self.emoji_neutral_halloween_label, self.emoji_sad_halloween_label, self.emoji_surprise_halloween_label]

    # panda emojis
        self.emoji_happy_panda = ImageTk.PhotoImage(Image.open("emojis/Happiness/happy_panda.png"))
        self.emoji_happy_panda_label = tk.Label(image=self.emoji_happy_panda, bg="#61808C")
        self.emoji_angry_panda = ImageTk.PhotoImage(Image.open("emojis/Anger/angry_panda.png"))
        self.emoji_angry_panda_label = tk.Label(image=self.emoji_angry_panda, bg="#61808C")
        self.emoji_disgust_panda = ImageTk.PhotoImage(Image.open("emojis/Disgust/disgust_panda.png"))
        self.emoji_disgust_panda_label = tk.Label(image=self.emoji_disgust_panda, bg="#61808C")
        self.emoji_fear_panda = ImageTk.PhotoImage(Image.open("emojis/Fear/fear_panda.png"))
        self.emoji_fear_panda_label = tk.Label(image=self.emoji_fear_panda, bg="#61808C")
        self.emoji_neutral_panda = ImageTk.PhotoImage(Image.open("emojis/Neutral/neutral_panda.png"))
        self.emoji_neutral_panda_label = tk.Label(image=self.emoji_neutral_panda, bg="#61808C")
        self.emoji_sad_panda = ImageTk.PhotoImage(Image.open('emojis/Sadness/sad_panda.png'))
        self.emoji_sad_panda_label = tk.Label(image=self.emoji_sad_panda, bg="#61808C")
        self.emoji_surprise_panda = ImageTk.PhotoImage(Image.open('emojis/Surprise/surprise_panda.png'))
        self.emoji_surprise_panda_label = tk.Label(image=self.emoji_surprise_panda, bg="#61808C")
    # array with panda emojis
        self.panda_ar = [self.emoji_angry_panda_label, self.emoji_disgust_panda_label, self.emoji_fear_panda_label, self.emoji_happy_panda_label, self.emoji_neutral_panda_label, self.emoji_sad_panda_label, self.emoji_surprise_panda_label]

    # strokeface emojis
        self.emoji_happy_strokeface = ImageTk.PhotoImage(Image.open('emojis/Happiness/happy_strokeface.png'))
        self.emoji_happy_strokeface_label = tk.Label(image=self.emoji_happy_strokeface, bg="#61808C")
        self.emoji_angry_strokeface = ImageTk.PhotoImage(Image.open('emojis/Anger/angry_strokeface.png'))
        self.emoji_angry_strokeface_label = tk.Label(image=self.emoji_angry_strokeface, bg="#61808C")
        self.emoji_disgust_strokeface = ImageTk.PhotoImage(Image.open('emojis/Disgust/disgust_strokeface.png'))
        self.emoji_disgust_strokeface_label = tk.Label(image=self.emoji_disgust_strokeface, bg="#61808C")
        self.emoji_fear_strokeface = ImageTk.PhotoImage(Image.open('emojis/Fear/fear_strokeface.png'))
        self.emoji_fear_strokeface_label = tk.Label(image=self.emoji_fear_strokeface, bg="#61808C")
        self.emoji_neutral_strokeface = ImageTk.PhotoImage(Image.open('emojis/Neutral/neutral_strokeface.png'))
        self.emoji_neutral_strokeface_label = tk.Label(image=self.emoji_neutral_strokeface, bg="#61808C")
        self.emoji_sad_strokeface = ImageTk.PhotoImage(Image.open('emojis/Sadness/sad_strokeface.png'))
        self.emoji_sad_strokeface_label = tk.Label(image=self.emoji_sad_strokeface, bg="#61808C")
        self.emoji_surprise_strokeface = ImageTk.PhotoImage(Image.open('emojis/Surprise/surprise_strokeface.png'))
        self.emoji_surprise_strokeface_label = tk.Label(image=self.emoji_surprise_strokeface, bg="#61808C")
    # array with strokeface emojis
        self.strokeface_ar = [self.emoji_angry_strokeface_label, self.emoji_disgust_strokeface_label, self.emoji_fear_strokeface_label, self.emoji_happy_strokeface_label, self.emoji_neutral_strokeface_label, self.emoji_sad_strokeface_label, self.emoji_surprise_strokeface_label]

    # emoji type icons used for radio buttons
        self.yellow_emoji_btn = tk.PhotoImage(file='buttons/yellow_emoji_button.png')
        self.basic_emoji_btn = tk.PhotoImage(file='buttons/basic_emoji_button.png')
        self.boy_emoji_btn = tk.PhotoImage(file='buttons/boy_emoji_button.png')
        self.halloween_emoji_btn = tk.PhotoImage(file='buttons/halloween_emoji_button.png')
        self.panda_emoji_btn = tk.PhotoImage(file='buttons/panda_emoji_button.png')
        self.strokeface_emoji_btn = tk.PhotoImage(file='buttons/strokeface_emoji_button.png')

    # close app button
        self.close_button = tk.Button(
            text="Close",
            width=25,
            height=5,
            bg="blue",
            fg="yellow",
            command=master.destroy
        )
        #self.close_button.place(x=1220, y=670)  # close app button
        self.close_button.place(x=750, y=600, relx=0.3, rely=0.08)

    # radio buttons for emoji type selection
        self.selected_emoji = tk.IntVar()
        self.selected_emoji.set(1)
        self.radio_btn1 = tk.Radiobutton(master, variable=self.selected_emoji, value=1, image=self.yellow_emoji_btn,
                                    command=lambda: self.update_emoji_type(0))
        self.radio_btn2 = tk.Radiobutton(master, variable=self.selected_emoji, value=2, image=self.basic_emoji_btn,
                                    command=lambda: self.update_emoji_type(1))
        self.radio_btn3 = tk.Radiobutton(master, variable=self.selected_emoji, value=3, image=self.boy_emoji_btn,
                                    command=lambda: self.update_emoji_type(2))
        self.radio_btn4 = tk.Radiobutton(master, variable=self.selected_emoji, value=4, image=self.halloween_emoji_btn,
                                    command=lambda: self.update_emoji_type(3))
        self.radio_btn5 = tk.Radiobutton(master, variable=self.selected_emoji, value=5, image=self.panda_emoji_btn,
                                    command=lambda: self.update_emoji_type(4))
        self.radio_btn6 = tk.Radiobutton(master, variable=self.selected_emoji, value=6, image=self.strokeface_emoji_btn,
                                    command=lambda: self.update_emoji_type(5))

    # placing of radio buttons
        #self.radio_btn1.place(x=1320, y=180)
        #self.radio_btn2.place(x=1320, y=240)
        #self.radio_btn3.place(x=1320, y=300)
        #self.radio_btn4.place(x=1320, y=360)
        #self.radio_btn5.place(x=1320, y=420)
        #self.radio_btn6.place(x=1320, y=480)
        self.radio_btn1.place(x=860, y=150, relx=0.3, rely=0.05)
        self.radio_btn2.place(x=860, y=210, relx=0.3, rely=0.05)
        self.radio_btn3.place(x=860, y=270, relx=0.3, rely=0.05)
        self.radio_btn4.place(x=860, y=330, relx=0.3, rely=0.05)
        self.radio_btn5.place(x=860, y=390, relx=0.3, rely=0.05)
        self.radio_btn6.place(x=860, y=450, relx=0.3, rely=0.05)

    # initial values of emoji type and current emoji
        self.current_emoji = self.yellow_ar[0]
        self.current_type = self.yellow_ar

# class functions-------------------------------------------------------------------------------------------------------

    def update_emoji_type(self, tp: int) -> None:
        '''
        Depending on the selected emoji-type, different emojis are displayed.
        There are 6 different types. The type is saved under 'current_type'.
        You can change the type by giving this method an input int from 0-5.
        :param tp: int from 0 to 5
        '''
        if tp == 0:
            self.current_type = self.yellow_ar
        elif tp == 1:
            self.current_type = self.basic_ar
        elif tp == 2:
            self.current_type = self.boy_ar
        elif tp == 3:
            self.current_type = self.halloween_ar
        elif tp == 4:
            self.current_type = self.panda_ar
        elif tp == 5:
            self.current_type = self.strokeface_ar

    def update_emoji(self, em: int) -> None:
        '''
        This function updates the currently displayed emotion-emoji.
        The emoji can be from a different type, depending what is currently written in 'current_type'.
        The emoji with the most fitting emotion should be displayed.
        The input is an int from 0-6 [angry, disgust, fear, happy, neutral, sad, surprise]
        :param em: int from 0 to 6
        '''
        if em == 0:
            self.current_emoji.place_forget()
            #self.current_type[0].place(x=875, y=205)
            self.current_type[0].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[0]
        elif em == 1:
            self.current_emoji.place_forget()
            #self.current_type[1].place(x=875, y=205)
            self.current_type[1].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[1]
        elif em == 2:
            self.current_emoji.place_forget()
            #self.current_type[2].place(x=875, y=205)
            self.current_type[2].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[2]
        elif em == 3:
            self.current_emoji.place_forget()
            #self.current_type[3].place(x=875, y=205)
            self.current_type[3].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[3]
        elif em == 4:
            self.current_emoji.place_forget()
            #self.current_type[4].place(x=875, y=205)
            self.current_type[4].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[4]
        elif em == 5:
            self.current_emoji.place_forget()
            #self.current_type[5].place(x=875, y=205)
            self.current_type[5].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[5]
        elif em == 6:
            self.current_emoji.place_forget()
            #self.current_type[6].place(x=875, y=205)
            self.current_type[6].place(x=580, y=170, relx=0.2, rely=0.05)
            self.current_emoji = self.current_type[6]

    def update_boarder(self, percentage: float) -> None:
        '''
        In order to visualize the correctness of the predicted emotion,
        this function updates the boarder color of the emoji window.
        If percentage < 0.5 -> red boarder
        If percentage > 0.5 -> green boarder
        :param percentage: float of the currently highest emotion percentage
        '''
        if (percentage < 0.5):
            self.emoji_view_green.place_forget()
            #self.emoji_view_red.place(x=850, y=180)
            self.emoji_view_red.place(x=550, y=140, relx=0.2, rely=0.05)
        else:
            self.emoji_view_red.place_forget()
            #self.emoji_view_green.place(x=850, y=180)
            self.emoji_view_green.place(x=550, y=140, relx=0.2, rely=0.05)

    def update_percentage(self, li: list) -> None:
        '''
        A list of all emojis are displayed on the gui.
        This function updates all values in percentage.
        :param li: output list of neural network, that contains values of the emotion labels
        '''
        self.anger.config(text="Anger: " + str(round(li[0] * 100)) + "%")
        self.disgust.config(text="Disgust: " + str(round(li[1] * 100)) + "%")
        self.fear.config(text="Fear: " + str(round(li[2] * 100)) + "%")
        self.happiness.config(text="Happiness: " + str(round(li[3] * 100)) + "%")
        self.neutral.config(text="Neutral: " + str(round(li[4] * 100)) + "%")
        self.sadness.config(text="Sadness: " + str(round(li[5] * 100)) + "%")
        self.surprise.config(text="Surprise: " + str(round(li[6] * 100)) + "%")

    def update_recognized_emotion(self, predicted_emotion: int) -> None:
        '''
        This function updates the image of the currently detected emotion-emoji.
        The input is an int from 0-6 [angry, disgust, fear, happy, neutral, sad, surprise]
        :param predicted_emotion: an int from 0-6 depending on the emotion
        '''
        if predicted_emotion == 0:
            self.detected_emotion.config(text="Detected Emotion: angry")
        elif predicted_emotion == 1:
            self.detected_emotion.config(text="Detected Emotion: disgust")
        elif predicted_emotion == 2:
            self.detected_emotion.config(text="Detected Emotion: fear")
        elif predicted_emotion == 3:
            self.detected_emotion.config(text="Detected Emotion: happy")
        elif predicted_emotion == 4:
            self.detected_emotion.config(text="Detected Emotion: neutral")
        elif predicted_emotion == 5:
            self.detected_emotion.config(text="Detected Emotion: sad")
        elif predicted_emotion == 6:
            self.detected_emotion.config(text="Detected Emotion: surprise")






