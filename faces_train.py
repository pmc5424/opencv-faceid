import cv2 as cv
import os
import numpy as np

people = []
DIR = r'/Users/ssraikhelkar/PycharmProjects/opencv-faceid/Faces/train'
for i in os.listdir(DIR):
    if i != '.DS_Store':
        people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray_img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray_img[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


if __name__ == '__main__':
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
