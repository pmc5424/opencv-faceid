import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    cv.imshow('Group Photo', img)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale Group Photo', gray_img)

    # Mask Creation
    blank = np.zeros(img.shape[:2], dtype='uint8')
    mask = cv.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
    cv.imshow('Circle', mask)

    masked_image = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('Masked Image', masked_image)

    # Compute Grayscale Histogram
    gray_hist = cv.calcHist([gray_img], [0], mask, [256], [0, 256])
    plt.figure()
    plt.title('Grayscale Hist')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

    # # Color Histogram
    # plt.figure()
    # plt.title('Color Hist')
    # plt.xlabel('Bins')
    # plt.ylabel('# of pixels')
    # colors = ('b', 'g', 'r')
    # for i, col in enumerate(colors):
    #     color_hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    #     plt.plot(color_hist)
    #     plt.xlim([0, 256])

    plt.show()

    cv.waitKey(0)
