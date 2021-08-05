import time
import cv2 as cv
import os
import numpy as np
import mediapipe as mp

DIR = os.path.join(os.getcwd(), 'Faces', 'train_mesh')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_mesh_trained.yml')

people = []


# Creates people list based on labeled data in the 'Faces/train' directory
def create_people():
    for i in os.listdir(DIR):
        if i != '.DS_Store':
            people.append(i)


def face_mesh_recognition_showcase():
    video_capture = cv.VideoCapture(0)
    time.sleep(1.0)

    while video_capture.isOpened():

        is_true, frame = video_capture.read()
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                face_canvas = imgRGB
                # face_canvas = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 3), dtype='uint8')
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

                if face_canvas_roi.size > 0:
                    gray_face_canvas_roi = cv.cvtColor(face_canvas_roi, cv.COLOR_RGB2GRAY)
                    label, confidence = face_recognizer.predict(gray_face_canvas_roi)
                    x, y = cx_min, cy_min

                    cv.putText(frame, people[label] + ' ' + str(round(confidence, 2)),
                               (x, y),
                               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                          drawSpec, drawSpec)

        cv.imshow('Video Capture', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


def face_mesh_recognition_authenticate():
    video_capture = cv.VideoCapture(0)
    time.sleep(1.0)

    label_counter = []
    for person in people:
        label_counter.append(0)

    while video_capture.isOpened():

        is_true, frame = video_capture.read()
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
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

                if face_canvas_roi.size > 0:
                    gray_face_canvas_roi = cv.cvtColor(face_canvas_roi, cv.COLOR_RGB2GRAY)
                    x, y = cx_min, cy_min

                    label, confidence = face_recognizer.predict(gray_face_canvas_roi)

                    if confidence < 1.00:
                        cv.putText(frame, people[label] + ' ' + str(round(confidence, 2)),
                                   (x, y),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                        mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                              drawSpec, drawSpec)
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
                            return

        cv.imshow('Video Capture', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    create_people()
    face_mesh_recognition_showcase()
    # face_mesh_recognition_authenticate()
