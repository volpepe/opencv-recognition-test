import numpy as np
import cv2

def show_image(image, name="Image"):
    # display the image to our screen -- we will need to click the window
    # open by OpenCV and press a key on our keyboard to continue execution
    cv2.imshow(name, image)
    cv2.waitKey(0)

def resize(image, newDimW, keep_ratio=False, newDimH=None):
    #resize an image without considering the aspect ratio
    h, w, d = image.shape
    if not keep_ratio:
        if not newDimH:
            newDimH = newDimW
        resized = cv2.resize(image, (newDimW, newDimH))
    else:
        #calculate the aspect ratio
        # h:w = x:200 --> x = 200*h/w
        new_height = newDimW * h / w
        resized = cv2.resize(image, (newDimW, int(new_height)))
    return resized

def main():
    #download trained object classifiers
    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')
    #glasses_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')

    img = cv2.imread('imgs/people.jpg')

    output = img.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.28, 5)
    for(x, y, w, h) in faces:
        #draw a rectangle on each face (x, x+w ...)
        output = cv2.rectangle(output, (x,y), (x+w,y+h), (255,0,0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = output[y:y+h, x:x+w]

        #detect smiles in each face roi
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=25)
        for(sx, sy, sw, sh) in smiles:
            #draw a rectangle inside the roi of the face
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 3)

        #detect glasses (broken)
        #glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=1)
        #for(gx, gy, gw, gh) in glasses:
        #    cv2.rectangle(roi_color, (gx,gy), (gx+gw,gy+gh), (0,255,0), 2)

        #detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.20, minNeighbors=7)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,255), 3)

    output = resize(output, 1000, keep_ratio=True)
    show_image(output, "Detection result")

if __name__ == "__main__":
    main()

