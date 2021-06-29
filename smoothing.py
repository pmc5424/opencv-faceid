import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('Photos/Group.jpeg')
    cv.imshow('Peake', img)

    # Averaging Blur
    avg_img = cv.blur(img, (3, 3))
    cv.imshow('Averaging Blur', avg_img)

    # Gaussian Blur
    gauss_img = cv.GaussianBlur(img, (3, 3), 0)
    cv.imshow('Gaussian Blur', gauss_img)

    # Median Blur
    med_img = cv.medianBlur(img, 3)
    cv.imshow('Median Blur', med_img)

    # Bilateral Blur
    bilateral_img = cv.bilateralFilter(img, 10, 35, 25)
    cv.imshow('Bilateral Blur', bilateral_img)

    cv.waitKey(0)
