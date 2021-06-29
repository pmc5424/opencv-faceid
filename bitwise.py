import cv2 as cv
import numpy as np


if __name__ == '__main__':
    blank_img = np.zeros((400, 400), dtype='uint8')

    rectangle = cv.rectangle(blank_img.copy(), (30, 30), (370, 370), 255, thickness=-1)
    circle = cv.circle(blank_img.copy(), (200, 200), 200, 255, thickness=-1)

    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)

    # and - only intersecting regions
    bitwise_and = cv.bitwise_and(rectangle, circle)
    cv.imshow('Bitwise And', bitwise_and)

    # or - both regions, including non-intersecting
    bitwise_or = cv.bitwise_or(rectangle, circle)
    cv.imshow('Bitwise Or', bitwise_or)

    # xor (eXclusive or) - non-intersecting regions
    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow('Bitwise Xclusive Or', bitwise_xor)

    # not - region
    bitwise_not = cv.bitwise_not(rectangle)
    cv.imshow('Bitwise Not', bitwise_not)

    cv.waitKey(0)
