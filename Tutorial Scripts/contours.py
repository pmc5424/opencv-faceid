import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('../Photos/Group.jpeg')
    cv.imshow('Group Photo', img)

    blank_img = np.zeros(img.shape, dtype='uint8')
    # cv.imshow('Blank', blank_img)

    # Grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Group Photo B/W', gray_img)

    # # Blur
    # blur = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
    # cv.imshow('Group Photo Blur', blur)
    #
    # # Canny
    # canny = cv.Canny(blur, 125, 175)
    # cv.imshow('Group Photo Canny', canny)

    ret, thresh = cv.threshold(gray_img, 125, 255, cv.THRESH_BINARY)
    # cv.imshow('Threshold Group Photo', thresh)

    # contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(f'{len(contours)} contour(s) found in the image')

    cv.drawContours(blank_img, contours, -1, (0, 0, 255), 1)
    cv.imshow('Contour Points', blank_img)

    cv.waitKey(0)
