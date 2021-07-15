import time
import cv2 as cv
import os

DIR = os.path.join(os.getcwd(), 'Faces', 'train')
haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = []


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


# Live feed of the facial recognition model in action
def face_recognition_showcase():
    video_capture = cv.VideoCapture(0)
    time.sleep(1.0)

    while video_capture.isOpened():
        is_true, frame = video_capture.read()

        label, (x, y, w, h), confidence = label_image(frame)

        if label != -1:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, people[label] + ' ' + str(round(confidence)), (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        cv.imshow('Video Capture', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


# Performs facial recognition until the face recognized has been labeled the same for at least 50 frames
# Outputs a string in the terminal to 'log in' with the label associated with the face
# TODO store images with eyes closed and recognize it to add security
def face_recognition_authenticate():
    video_capture = cv.VideoCapture(0)
    time.sleep(1.0)

    label_counter = []
    for person in people:
        label_counter.append(0)

    while video_capture.isOpened():
        is_true, frame = video_capture.read()

        label, (x, y, w, h), confidence = label_image(frame)

        if label != -1 and confidence < 40:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, people[label], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            label_counter[label] += 1

            max_label_count = 0
            max_label = 0

            for i in range(len(label_counter)):
                label_count = label_counter[i]
                if label_count > max_label_count:
                    max_label_count = label_count
                    max_label = i

            if max_label_count > 50:
                cv.destroyAllWindows()
                print(f'You have been logged in as {people[max_label]}.')
                break

        cv.imshow('Video Capture', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


# # Tests accuracy of the facial recognition model using all the labeled images in the 'Faces/val' directory
# # ONLY WORKS WHEN ALL LABELS USED IN TRAINING HAVE AT LEAST 1 IMAGE FOR VALIDATION
# def test_all_val():
#     val_dir = os.path.join(DIR, 'val')
#     num_correct = 0
#     num_total = 0
#     label_counter = np.zeros(len(people))
#
#     for person in people:
#         path = os.path.join(val_dir, person)
#
#         img_list = os.listdir(path)
#
#         for img in img_list:
#             if img != '.DS_Store':
#                 img = cv.imread(os.path.join(path, img))
#                 label, face_rect = label_image(img)
#                 if people[label] == person:
#                     num_correct += 1
#                     label_counter[label] += 1
#                 num_total += 1
#
#     print(f'Model was correct {num_correct} times and had {num_correct} / {num_total} or'
#           f' {round(num_correct / num_total * 100, 3)}% accuracy.')
#     print()
#     for person in people:
#         print(f'{person} was accurately recognized {label_counter[people.index(person)]} times')


if __name__ == '__main__':
    create_people()
    face_recognition_showcase()
    # face_recognition_authenticate()
