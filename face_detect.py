import cv2 as cv


if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    # cv.imshow('Photo', img)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Grayscale', gray_img)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    # faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8)
    # print(len(faces_rect))
    #
    # for (x, y, w, h) in faces_rect:
    #     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #
    # cv.imshow('Detected Faces', img)

    vid_capture = cv.VideoCapture(0)
    while True:
        is_true, frame = vid_capture.read()

        faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)
        for (x, y, w, h) in faces_rect:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Video Capture', frame)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
