import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('../Photos/Peake.jpeg')
    cv.imshow('Peake', img)

    blank_img = np.zeros(img.shape[:2], dtype='uint8')

    b, g, r = cv.split(img)

    blue_img = cv.merge([b, blank_img, blank_img])
    green_img = cv.merge([blank_img, g, blank_img])
    red_img = cv.merge([blank_img, blank_img, r])
    cv.imshow('Blue', blue_img)
    cv.imshow('Green', green_img)
    cv.imshow('Red', red_img)

    # cv.imshow('Blue', b)
    # cv.imshow('Green', g)
    # cv.imshow('Red', r)

    merged_img = cv.merge([b, g, r])
    # cv.imshow('Merged', merged_img)



    cv.waitKey(0)
