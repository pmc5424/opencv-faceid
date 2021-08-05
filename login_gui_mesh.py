import time
import cv2 as cv
import PySimpleGUI as sg
import os
import numpy as np
import mediapipe as mp

DIR = os.path.join(os.getcwd(), 'Faces', 'train_mesh')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

features = []
labels = []
people = []
face_recognizer = cv.face.LBPHFaceRecognizer_create()


# Creates training data for face recognition model
def create_train():
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

            global features, labels
            features.append(gray_img)
            labels.append(label)


# Creates people list based on labeled data in the 'Faces/train' directory
def create_people():
    for i in os.listdir(DIR):
        if i != '.DS_Store':
            people.append(i)


# Set a theme for the GUIs
sg.theme('Black')

# Create the various layouts for the different GUIs

login_layout = [[sg.Text("Log in to Presence Browser", auto_size_text=True, key='-LOGIN-TEXT-')],
                [sg.Text("Enter your full name: "), sg.InputText('', enable_events=True, key='-DISPLAY-NAME-')],
                [sg.Button("Login", bind_return_key=True, pad=(5, 10))]]

new_user_layout = [[sg.Text("Direct your face toward the camera. Press Next when you're ready.                        "
                            "       ",
                            auto_size_text=True, key='-PROMPT-TEXT-')],
                   [sg.Image(filename="", key="-IMAGE-")],
                   [sg.Button("Next", size=(10, 1), visible=True, bind_return_key=True, key='-NEXT-')]]

authenticate_layout = [[sg.Text("Come close and direct your face to the camera to log in.", key='-PROMPT-TEXT-')],
                       [sg.Image(filename="", key="-IMAGE-")],
                       [sg.Button("Login", size=(10, 1), visible=False)]]


# Login function that opens a login GUI and retrieves the user's desired DisplayName
def login():
    login_window = sg.Window("Presence Browser Login", login_layout, size=(400, 120),
                             return_keyboard_events=True).Finalize()
    login_window.maximize()

    while True:
        event, values = login_window.read()

        if event == sg.WIN_CLOSED:
            login_window.close()
            return -1, ''

        # If the user presses the 'Login' button
        if event == "Login":
            name = values['-DISPLAY-NAME-']

            # Check if the model has already been trained on 'name' and authenticate if so
            if not os.path.isdir(os.path.join(os.getcwd(), 'Faces', 'train_mesh', name)):
                login_window.close()
                return 0, name

            # If model not trained on 'name', capture new images and train model on new face
            else:
                login_window.close()
                return 1, name


# Function that stores faces labeled as name to retrain the face recognizer for a new user
def new_user_create(name):
    # Start webcam and wait for warm-up
    vid_cap = cv.VideoCapture(0)
    time.sleep(1.00)

    new_user_create_window = sg.Window("Create New User", new_user_layout, location=(0, 0),
                                       element_justification='c').Finalize()
    new_user_create_window.maximize()

    # List of prompts to be displayed to the user
    prompts = ["Come close and direct your face toward the camera. Press Next when you're ready.                       "
               "      ",
               "Now slightly tilt your face to the right and face the camera. Press Next when ready.",
               "Again slightly tilt your face but to the left. Press Next when you're ready.",
               "Done! Now press next to train the facial recognition model.",
               "Finished setup! Restart and log in with your full name."]
    prompt = 0

    os.mkdir(os.path.join(os.getcwd(), 'Faces', 'train_mesh', name))
    curr_photo_num = 1
    curr_face_canvas_roi = None
    BRIGHTNESS = 160.0

    while True:
        event, values = new_user_create_window.read(timeout=0)

        # Read from web cam and get the frame being captured
        ret, frame = vid_cap.read()

        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # TODO
                # face_canvas = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 3), dtype='uint8')
                face_canvas = imgRGB
                mpDraw.draw_landmarks(face_canvas, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                      drawSpec, drawSpec)

                h, w, c = face_canvas.shape
                cx_min = w
                cy_min = h
                cx_max = cy_max = 0

                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if cx < cx_min:
                        cx_min = cx
                    if cy < cy_min:
                        cy_min = cy
                    if cx > cx_max:
                        cx_max = cx
                    if cy > cy_max:
                        cy_max = cy

                face_canvas_roi = face_canvas[cy_min:cy_max, cx_min:cx_max]

                if face_canvas_roi.size > 0 and (cx_max - cx_min) * (cy_max - cy_min) > 0.12 * (
                        imgRGB.shape[0] * imgRGB.shape[1]):
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                          drawSpec, drawSpec)
                    curr_face_canvas_roi = face_canvas_roi

        img = cv.imencode(".png", frame)[1].tobytes()
        new_user_create_window.FindElement("-IMAGE-").update(data=img)

        # Save region of frame with a face to training directory labeled by user's name
        if not new_user_create_window.FindElement(
                '-NEXT-').visible and curr_photo_num <= 60 and curr_face_canvas_roi is not None:
            gray_face_canvas_roi = cv.cvtColor(curr_face_canvas_roi, cv.COLOR_RGB2GRAY)
            curr_face_canvas_roi = None
            filename = f'mesh_{curr_photo_num}'
            cv.imwrite(f'{os.path.join(os.getcwd(), "Faces", "train_mesh", name, filename)}.png',
                       gray_face_canvas_roi)
            curr_photo_num += 1

        if curr_photo_num % 20 == 0:
            new_user_create_window.FindElement('-NEXT-').update(visible=True)
            new_user_create_window.FindElement('-PROMPT-TEXT-').update(visible=True)

        # Show the next prompt when 'Next' is clicked
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
                face_recognizer.save('face_mesh_trained.yml')
                np.save('features_mesh.npy', features)
                np.save('labels_mesh.npy', labels)

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
    # Start web cam and wait for warm-up
    vid_cap = cv.VideoCapture(0)
    time.sleep(1.00)

    authenticate_window = sg.Window(f"Login as {name}", authenticate_layout, location=(0, 0),
                                    element_justification='c').Finalize()
    authenticate_window.maximize()

    login_label = people.index(name)
    label_count = 0
    BRIGHTNESS = 160.0

    while True:
        event, values = authenticate_window.read(timeout=0)

        # Read from the web cam and capture frames
        ret, frame = vid_cap.read()

        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # TODO
                # face_canvas = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 3), dtype='uint8')
                face_canvas = imgRGB
                mpDraw.draw_landmarks(face_canvas, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                      drawSpec, drawSpec)

                h, w, c = face_canvas.shape
                cx_min = w
                cy_min = h
                cx_max = cy_max = 0

                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if cx < cx_min:
                        cx_min = cx
                    if cy < cy_min:
                        cy_min = cy
                    if cx > cx_max:
                        cx_max = cx
                    if cy > cy_max:
                        cy_max = cy

                face_canvas_roi = face_canvas[cy_min:cy_max, cx_min:cx_max]

                if face_canvas_roi.size > 0 and (cx_max - cx_min) * (cy_max - cy_min) > 0.12 * (
                        imgRGB.shape[0] * imgRGB.shape[1]):
                    gray_face_canvas_roi = cv.cvtColor(face_canvas_roi, cv.COLOR_RGB2GRAY)
                    x, y = cx_min, cy_min

                    label, confidence = face_recognizer.predict(gray_face_canvas_roi)

                    if label == login_label and confidence < confidence_value:
                        cv.putText(frame, people[label],
                                   (x, y),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                        mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                              drawSpec, drawSpec)
                        label_count += 1

        img = cv.imencode(".png", frame)[1].tobytes()
        authenticate_window.FindElement("-IMAGE-").update(data=img)

        # Once label count incremented at least 50 times, display login button and success prompt
        if label_count > 50:
            authenticate_window.FindElement('-PROMPT-TEXT-').update(f'Identified as {name}')
            authenticate_window.FindElement('Login').update(visible=True)

        # Close window and web cam when 'Login' clicked
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
        face_recognizer.read('face_mesh_trained.yml')
        authenticate(display_name, 30)
