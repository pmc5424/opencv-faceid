import time
import cv2 as cv
import os
import numpy as np

DIR = r'/Users/ssraikhelkar/PycharmProjects/opencv-faceid/Faces'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = []
empty = ()


def create_people():
    for i in os.listdir(os.path.join(DIR, 'train')):
        if i != '.DS_Store':
            people.append(i)


def label_image(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)
    if faces_rect is not empty:
        for (x, y, w, h) in faces_rect:
            face_roi = gray_img[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(face_roi)
            return label, (x, y, w, h)
    return -1, (0, 0, 0, 0)


def face_recognition_display():
    video_capture = cv.VideoCapture(0)
    time.sleep(1.0)

    while video_capture.isOpened():
        is_true, frame = video_capture.read()

        label, (x, y, w, h) = label_image(frame)

        if label != -1:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, people[label], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

            cv.imshow('Video Capture', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def test_all_val():
    val_dir = os.path.join(DIR, 'val')
    num_correct = 0
    num_total = 0
    label_counter = np.zeros(len(people))

    for person in people:
        path = os.path.join(val_dir, person)

        img_list = os.listdir(path)

        for img in img_list:
            if img != '.DS_Store':
                img = cv.imread(os.path.join(path, img))
                label, face_rect = label_image(img)
                if people[label] == person:
                    num_correct += 1
                    label_counter[label] += 1
                num_total += 1

    print(f'Model was correct {num_correct} times and had {num_correct} / {num_total} or'
          f' {round(num_correct / num_total * 100, 3)}% accuracy.')
    print()
    for person in people:
        print(f'{person} was accurately recognized {label_counter[people.index(person)]} times')


if __name__ == '__main__':
    create_people()
    face_recognition_display()
    # test_all_val()
