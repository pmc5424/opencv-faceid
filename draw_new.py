import cv2 as cv
import numpy as np

if __name__ == '__main__':
    blank_img = np.zeros((500, 500, 3), dtype='uint8')
    cv.imshow('Blank', blank_img)

    for i in range(500):
        for j in range(500):
            if (i + j) % 100 == 0:
                blank_img[i, j] = 255, 255, 255

    cv.rectangle(blank_img, (200, 200), (400, 300), color=(255, 0, 0), thickness=-1)
    cv.line(blank_img, (0, 0), (200, 300), (255, 255, 255), thickness=5)
    cv.circle(blank_img, (blank_img.shape[0] // 2, blank_img.shape[1] // 2), 20, color=(0, 255, 0), thickness=-1)
    cv.circle(blank_img, (blank_img.shape[0] // 2 + 5, blank_img.shape[1] // 2 - 5), 15, color=(255, 0, 0),
              thickness=-1)

    cv.putText(blank_img, 'Dort MVP', (blank_img.shape[0] // 3, blank_img.shape[1] // 3),
               fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(0, 0, 255))


    cv.imshow('Drawing', blank_img)
    cv.waitKey(0)
