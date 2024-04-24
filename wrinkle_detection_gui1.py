import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.models import model_from_json


def WrinkleModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

window = tk.Tk()
window.geometry('800x600')
window.title('Wrinkle Detector')
window.configure(background='#CDCDCD')

label_result = Label(window, background='#CDCDCD', font=('arial', 15, 'bold'))
label_image = Label(window)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = WrinkleModel("model_wrinkle.json", "model_weights_wrinkle.h5")

def detect_wrinkles(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) > 0:
        label_result.configure(foreground="#011638", text="Wrinkles Detected")
    else:
        label_result.configure(foreground="#011638", text="No Wrinkles Detected")

def show_detect_button(file_path):
    detect_button = Button(window, text="Detect Wrinkles", command=lambda: detect_wrinkles(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        uploaded_image.thumbnail(((window.winfo_width() / 2.25), (window.winfo_height() / 2.25)))
        image = ImageTk.PhotoImage(uploaded_image)

        label_image.configure(image=image)
        label_image.image = image
        label_result.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(e)

upload_button = Button(window, text="Upload Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_button.pack(side='bottom', pady=50)

label_image.pack(side='bottom', expand='True')
label_result.pack(side='bottom', expand='True')

heading = Label(window, text='Wrinkle Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

window.mainloop()
