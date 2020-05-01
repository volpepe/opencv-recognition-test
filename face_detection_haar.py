import numpy as np
import cv2

#download trained object classifiers
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')
glasses_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')

img = cv2.imread('imgs/people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces) 

