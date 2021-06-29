import cv2 as cv
import numpy as np


def translate(img, x, y):
    trans_mat = np.float32(([1, 0, x], [0, 1, y]))
    dimensions = (img.shape[1], img.shape[0])

    return cv.warpAffine(img, trans_mat, dimensions)


def rotate(img, angle, rot_pt=None):
    (height, width) = img.shape[:2]

    if rot_pt is None:
        rot_pt = (width // 2, height // 2)

    rot_mat = cv.getRotationMatrix2D(rot_pt, angle, 1.0)
    dimensions = (img.shape[1], img.shape[0])

    return cv.warpAffine(img, rot_mat, dimensions)


if __name__ == '__main__':
    dort_img = cv.imread('Photos/Dort.png')
    cv.imshow('Dort OG', dort_img)

    # Translate
    cv.imshow('Dort Translated', translate(dort_img, 50, 50))

    # Rotate
    cv.imshow('Dort Rotated', rotate(dort_img, -45, (0, 0)))

    # Resize
    cv.imshow('Dort Resized', cv.resize(dort_img, (500, 500), cv.INTER_CUBIC))

    # Reflection
    cv.imshow('Dort Reflected', cv.flip(dort_img, 1))

    # Cropping
    cv.imshow('Dort Cropped', dort_img[0:100, 100:200])

    cv.waitKey(0)
