import PySimpleGUI as sg
import cv2 as cv

sg.theme('DarkBlue1')

layout = [[sg.Text("OpenCV Test")],
          [sg.Image(filename="", key="-IMAGE-")],
          [sg.Button("Next", size=(10, 1), visible=False)]]

window = sg.Window("OpenCVTest", layout, size=(800, 800))
capture = cv.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    ret, frame = capture.read()

    img = cv.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=img)

capture.release()
window.close()
