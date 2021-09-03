# Facial Recognition SSO
A proof of concept of a Facial Recogniton-based SSO for the Presence Browser project using OpenCV and MediaPipe Face Mesh.


Utilizes the MediaPipe Face Mesh API to identify faces in a webcam's video feed and then capture an image of their face and its key points.

Grayscaled Face Image with Key Points from Face Mesh API used for Model Training

[![Sample Training Image](https://i.postimg.cc/NjTvZGMn/image.png)](https://postimg.cc/G4cfyCvP)


After capturing a set of 60 images, an LBPH facial recognition model is then trained on these images and previously captured ones to estimate the binarized histogram of one's face. With this information, the login-script searches for a particular label for at least 50 frames to then allow a user to log-in.
