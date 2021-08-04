import cv2 as cv


def webcam_feed():
    vid = cv.VideoCapture(0)

    while True:
        is_true, frame = vid.read()
        # grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # blurred = cv.GaussianBlur(frame, (7, 7), cv.BORDER_DEFAULT)
        # canny = cv.Canny(frame, 125, 225)
        # dilated = cv.dilate(canny, (7, 7), iterations=2)

        cv.imshow('Video Feed', frame[40:800, 50:700])

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()


def image_reader(player_name):
    player_img = cv.imread('Photos/' + player_name)
    cv.imshow(player_name, player_img)

    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()


def video_reader():
    player_video = cv.VideoCapture('Videos/Dort.mp4')

    while True:
        is_true, frame = player_video.read()
        cv.imshow('Dort Video', frame)

        frame_resized = frame_resizer(frame, scale=1.5)
        cv.imshow('Dort Video Scaled', frame_resized)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


def frame_resizer(frame, scale=0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = width, height

    return cv.resize(frame, dimensions, cv.INTER_AREA)


if __name__ == '__main__':
    webcam_feed()
    # image_reader('Muscala.png')
    # video_reader()
