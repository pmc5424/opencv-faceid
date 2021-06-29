import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    # cv.imshow('Group', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    # Laplacian
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv.imshow('Lap', lap)

    # Sobel
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)

    # cv.imshow('Sobel X', sobel_x)
    # cv.imshow('Sobel Y', sobel_y)

    combined_sobel = cv.bitwise_or(sobel_x, sobel_y)
    cv.imshow('Combined Sobel', combined_sobel)

    # Canny
    canny = cv.Canny(gray, 150, 175)
    cv.imshow('Canny', canny)


    cv.waitKey(0)
