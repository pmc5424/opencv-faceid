import time
import cv2 as cv
import PySimpleGUI as sg
import os

import numpy
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')
DIR = os.path.join(os.getcwd(), 'Faces', 'train')
features = []
labels = []
people = []
face_recognizer = cv.face.LBPHFaceRecognizer_create()


# Creates training data for face recognition model
def create_train():
    global people
    for i in os.listdir(DIR):
        if i != '.DS_Store':
            people.append(i)

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray_img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray_img[y:y + h, x:x + w]
                global features, labels
                features.append(faces_roi)
                labels.append(label)


# Creates people list based on labeled data in the 'Faces/train' directory
def create_people():
    for i in os.listdir(DIR):
        if i != '.DS_Store':
            people.append(i)


# Labels a passed in opencv image and returns a corresponding label and facial coordinates tuple
def label_image(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    empty = ()

    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)

    if faces_rect is not empty:
        for (x, y, w, h) in faces_rect:
            face_roi = gray_img[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(face_roi)
            return label, (x, y, w, h), confidence
    return -1, (0, 0, 0, 0), 0


# Set a theme for the GUIs
sg.theme('DarkBlue17')

# Create the various layouts for the different GUIs

login_layout = [[sg.Text("Log in to Presence Browser", auto_size_text=True, key='-LOGIN-TEXT-')],
                [sg.Text("Enter your full name: "), sg.InputText('', enable_events=True, key='-DISPLAY-NAME-')],
                [sg.Button("Login", bind_return_key=True, pad=(5, 10))]]

new_user_layout = [[sg.Text("Direct your face toward the camera. Press Next when you're ready.                        "
                            "       ",
                            auto_size_text=True, key='-PROMPT-TEXT-')],
                   [sg.Image(filename="", key="-IMAGE-")],
                   [sg.Button("Next", size=(10, 1), visible=True, bind_return_key=True, key='-NEXT-')]]

authenticate_layout = [[sg.Text("Direct your face to the camera to log in.", key='-PROMPT-TEXT-')],
                       [sg.Image(filename="", key="-IMAGE-")],
                       [sg.Button("Login", size=(10, 1), visible=False)]]


# Login function that opens a login GUI and retrieves the user's desired DisplayName
# TODO add delete option that deletes the images associated with a user and retrains the facial recognition model
def login():
    login_window = sg.Window("Presence Browser Login", login_layout, size=(400, 120), return_keyboard_events=True)

    while True:
        event, values = login_window.read()

        if event == sg.WIN_CLOSED:
            login_window.close()
            return -1, ''

        # If the user presses the 'Login' button
        if event == "Login":
            name = values['-DISPLAY-NAME-']

            # Check if the model has already been trained on 'name' and authenticate if so
            if not os.path.isdir(os.path.join(os.getcwd(), 'Faces', 'train', name)):
                login_window.close()
                return 0, name

            # If model not trained on 'name', capture new images and train model on new face
            else:
                login_window.close()
                return 1, name


# Function that stores faces labeled as name to retrain the face recognizer for a new user
# TODO make sure a 'new user' does not have a face already existing in the database
def new_user_create(name):
    vid_cap = cv.VideoCapture(0)
    time.sleep(1.00)

    new_user_create_window = sg.Window("Create New User", new_user_layout, size=(800, 800), element_justification='c')

    prompts = ["Direct your face toward the camera. Press Next when you're ready.                                     "
               "      ",
               "Now slightly tilt your face to the right and face the camera. Press Next when you're ready.",
               "Again slightly tilt your face but to the left. Press Next when you're ready.",
               "Done! Now press next to train the facial recognition model.",
               "Finished setup! Restart and log in with your full name."]
    prompt = 0
    os.mkdir(os.path.join(os.getcwd(), 'Faces', 'train', name))
    curr_photo_num = 1
    BRIGHTNESS = 160.0

    while True:
        event, values = new_user_create_window.read(timeout=0)

        ret, frame = vid_cap.read()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        curr_brightness = np.average(frame[:, :, 2])
        brightness_diff = BRIGHTNESS - curr_brightness
        if brightness_diff < 50:
            frame[:, :, 2] = np.clip(frame[:, :, 2] + brightness_diff, 0, 255)
        frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)

        faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces_rect:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        img = cv.imencode(".png", frame)[1].tobytes()
        new_user_create_window.FindElement("-IMAGE-").update(data=img)

        if not new_user_create_window.FindElement('-NEXT-').visible and curr_photo_num <= 60:
            cv.imwrite(f'{os.path.join(os.getcwd(), "Faces", "train", name, str(curr_photo_num))}.jpg',
                       gray[y:y + h, x:x + w])
            curr_photo_num += 1

        if curr_photo_num % 20 == 0:
            new_user_create_window.FindElement('-NEXT-').update(visible=True)
            new_user_create_window.FindElement('-PROMPT-TEXT-').update(visible=True)

        if event == "-NEXT-":
            prompt += 1

            if prompt <= 3:
                new_user_create_window.FindElement('-PROMPT-TEXT-').update(visible=False)
                new_user_create_window.FindElement('-NEXT-').update(visible=False)

            # Perform training once all necessary training images are stored
            elif prompt == len(prompts) - 1:
                create_train()

                global features, labels, face_recognizer
                features = np.array(features, dtype='object')
                labels = np.array(labels)

                # Train recognizer
                face_recognizer.train(features, labels)
                face_recognizer.save('face_trained.yml')
                np.save('features.npy', features)
                np.save('labels.npy', labels)

                new_user_create_window.FindElement('-PROMPT-TEXT-').update(visible=True)
                new_user_create_window.FindElement('-PROMPT-TEXT-').update('Model trained successfully.')
                new_user_create_window.FindElement('-NEXT-').update(visible=True)

            # Exit once all steps of set-up have been completed
            elif prompt == len(prompts):
                new_user_create_window.close()
                return 1

            else:
                new_user_create_window.FindElement('-NEXT-').update(visible=True)
                new_user_create_window.FindElement('-PROMPT-TEXT-').update(visible=True)

        # Update prompt text when 'Next' button is pressed
        new_user_create_window.FindElement('-PROMPT-TEXT-').update(prompts[prompt])

        if event == sg.WIN_CLOSED:
            new_user_create_window.close()
            return -1


# Function that verifies whether a user logging in has a face that matches how the face recognizer labels the
# already existing images from training
def authenticate(name, confidence_value):
    vid_cap = cv.VideoCapture(0)
    time.sleep(1.00)

    authenticate_window = sg.Window(f"Login as {name}", authenticate_layout, size=(800, 800),
                                    element_justification='c')

    login_label = people.index(name)
    label_count = 0
    BRIGHTNESS = 160.0

    while True:
        event, values = authenticate_window.read(timeout=0)

        ret, frame = vid_cap.read()
        label, (x, y, w, h), confidence = label_image(frame)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        curr_brightness = np.average(frame[:, :, 2])
        brightness_diff = BRIGHTNESS - curr_brightness
        if brightness_diff < 50:
            frame[:, :, 2] = np.clip(frame[:, :, 2] + brightness_diff, 0, 255)
        frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)

        if label == login_label and confidence < confidence_value:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            label_count += 1

        img = cv.imencode(".png", frame)[1].tobytes()
        authenticate_window.FindElement("-IMAGE-").update(data=img)

        if label_count > 50:
            authenticate_window.FindElement('-PROMPT-TEXT-').update(f'Identified as {name}')
            authenticate_window.FindElement('Login').update(visible=True)

        if event == 'Login':
            authenticate_window.close()
            vid_cap.release()
            print(f'Successfully logged in as {name}')
            return 1

        if event == sg.WIN_CLOSED:
            authenticate_window.close()
            vid_cap.release()
            return -1


if __name__ == '__main__':
    request_code, display_name = login()

    # If name does not have any faces associated with it, store new images of their face
    if request_code == 0:
        new_user_create(display_name)

    # If an existing user, load the facial recognition model and authenticate using their face
    if request_code == 1:
        create_people()
        face_recognizer.read('face_trained.yml')
        authenticate(display_name, 50)
