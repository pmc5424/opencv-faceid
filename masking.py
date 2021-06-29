import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    cv.imshow('Group Photo', img)

    # Mask must be same size as img shape
    blank = np.zeros(img.shape[:2], dtype='uint8')
    # cv.imshow('Blank', blank)

    mask = cv.circle(blank.copy(), (20 + img.shape[1] // 2, -300 + img.shape[0] // 2), 100, 255, -1)
    # cv.imshow('Mask', mask)

    masked_img = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('Masked Image', masked_img)

    cv.waitKey(0)
