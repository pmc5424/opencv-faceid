import os
import shutil
import numpy as np
import cv2 as cv
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


name = ''
DIR = os.path.join(os.getcwd(), os.path.join('Faces', 'train_mesh'))
features = []
labels = []


# Creates training data for face recognition model
def create_train():
    people = []
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

            features.append(gray_img)
            labels.append(label)


# Captures and stores faces in grayscale to a directory labeled with a name for training
def face_frame_capture(curr_photo_num, prompt):
    video_capture = cv.VideoCapture(0)
    final_photo_num = curr_photo_num + 20

    while curr_photo_num < final_photo_num:
        is_true, frame = video_capture.read()

        if prompt == 0:
            cv.putText(frame, 'Direct your face directly toward the camera',
                       (20, frame.shape[0] // 2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))
        elif prompt == 1:
            cv.putText(frame, 'Direct your face to the right to show the left side of your face',
                       (20, frame.shape[0] // 2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))
        else:
            cv.putText(frame, 'Direct your face to the left to show the right side of your face',
                       (20, frame.shape[0] // 2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))

        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                face_landmarks = faceLms
                face_canvas = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 3), dtype='uint8')
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

                if face_canvas_roi is not None or face_canvas_roi is not []:
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                          drawSpec, drawSpec)
                    gray_face_canvas_roi = cv.cvtColor(face_canvas_roi, cv.COLOR_RGB2GRAY)
                    filename = f'mesh_{curr_photo_num}'
                    cv.imwrite(f'{os.path.join(os.getcwd(), "Faces", "train_mesh", name, filename)}.png',
                               gray_face_canvas_roi)
                    curr_photo_num += 1

        cv.imshow('Video Capture', frame)
        cv.waitKey(20)


if __name__ == '__main__':
    name = input('Enter tne name to associate with your face (<First Name> <Last Name>): ')

    if not os.path.isdir(os.path.join('Faces', 'train_mesh', name)):
        os.mkdir(os.path.join('Faces', 'train_mesh', name))

        input('Direct your face directly to the camera. Press Enter when ready for capturing...')
        face_frame_capture(0, 0)
        print('Done!')
        input('Now move your face slightly to the right of the camera. Pres Enter when ready... ')
        face_frame_capture(20, 1)
        print('Done!')
        input('Now move your face slightly to the left of the camera. Press Enter when ready...')
        face_frame_capture(40, -1)
        print('Done! Your face is now ready to be used for authentication.')

        print('---------Performing training-------------')
        create_train()

        face_recognizer = cv.face.LBPHFaceRecognizer_create()

        features = np.array(features, dtype='object')
        labels = np.array(labels)

        # Train recognizer
        face_recognizer.train(features, labels)
        face_recognizer.save('face_mesh_trained.yml')
        np.save('features_mesh.npy', features)
        np.save('labels_mesh.npy', labels)
        print(f'----------------Model trained using {len(labels)} labelled photos---------------------')

    else:
        choice = input('Your face is already stored. Delete and retrain model without face? <y/n>: ')

        if choice == 'y':
            shutil.rmtree(os.path.join('Faces', 'train', name))

            print('---------Performing training-------------')
            create_train()

            face_recognizer = cv.face.LBPHFaceRecognizer_create()

            features = np.array(features, dtype='object')
            labels = np.array(labels)

            # Train recognizer
            face_recognizer.train(features, labels)
            face_recognizer.save('face_mesh_trained.yml')
            np.save('features_mesh.npy', features)
            np.save('labels_mesh.npy', labels)
            print(f'----------------Model trained using {len(labels)} labelled photos---------------------')
