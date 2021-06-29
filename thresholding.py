import cv2 as cv
import numpy as np


if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    # cv.imshow('Group', img)

    blank = np.zeros(img.shape, dtype='uint8')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Group B/W', gray)

    # Simple Thresholding
    threshold, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    cv.imshow('Thresh Img', thresh)

    threshold, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Thresh Inv Img', thresh_inv)

    # Adaptive Threshold
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 8)
    cv.imshow('Adaptive Thresh Img', adaptive_thresh)


    cv.waitKey(0)
