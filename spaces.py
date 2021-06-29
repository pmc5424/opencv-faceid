import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    cv.imshow('Photo', img)

    # BGR to Grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray_img)

    # BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('HSV', hsv_img)

    # BGR to LAB
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cv.imshow('LAB', lab_img)

    # HSV to BGR
    hsv_bgr = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    cv.imshow('HSV-BGR', hsv_bgr)

    # LAB to BGR
    lab_bgr = cv.cvtColor(lab_img, cv.COLOR_LAB2BGR)
    cv.imshow('LAB-BGR', lab_bgr)

    # Gray to BGR
    gray_bgr = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
    cv.imshow('Gray-BGR', gray_bgr)

    cv.waitKey(0)
