import os
import shutil
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
name = ''
DIR = os.path.join(os.getcwd(), os.path.join('Faces', 'train'))
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

            faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray_img[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


# Captures and stores faces in grayscale to a directory labeled with a name for training
def face_frame_capture(curr_photo_num, prompt):
    video_capture = cv.VideoCapture(0)

    final_photo_num = curr_photo_num + 20

    while curr_photo_num < final_photo_num:
        is_true, frame = video_capture.read()

        if prompt == 0:
            cv.putText(frame, 'Direct your face directly toward the camera',
                       (20, frame.shape[0]//2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))
        elif prompt == 1:
            cv.putText(frame, 'Direct your face to the right to show the left side of your face',
                       (20, frame.shape[0] // 2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))
        else:
            cv.putText(frame, 'Direct your face to the left to show the right side of your face',
                       (20, frame.shape[0] // 2 - 300), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.0, color=(0, 0, 0))

        faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces_rect:
            cv.imwrite(f'{os.path.join(os.getcwd(), "Faces", "train", name, str(curr_photo_num))}.jpg',
                       gray[y:y + h, x:x + w])
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            curr_photo_num += 1

        cv.imshow('Video Capture', frame)
        cv.waitKey(20)


if __name__ == '__main__':

    name = input('Enter tne name to associate with your face (<First Name> <Last Name>): ')

    if not os.path.isdir(os.path.join('Faces', 'train', name)):
        os.mkdir(os.path.join('Faces', 'train', name))

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
        face_recognizer.save('face_trained.yml')
        np.save('features.npy', features)
        np.save('labels.npy', labels)
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
            face_recognizer.save('face_trained.yml')
            np.save('features.npy', features)
            np.save('labels.npy', labels)
            print(f'----------------Model trained using {len(labels)} labelled photos---------------------')
