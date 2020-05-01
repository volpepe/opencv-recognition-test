# import the necessary packages
import imutils
import cv2
import numpy as np

def show_image(image, name="Image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)

image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
show_image(image)

#get RGB values of a pixel
(B, G, R) = image[100, 50] #x 50, y 100
print("R={}, G={}, B={}".format(R, G, B))

#change the image so that the red channel is 0
image_copy = np.array(image, copy=True)
for hi in range(h):
    for wi in range(w):
        (B, G, R) = image[hi, wi]
        image_copy[hi, wi] = (B, G, 0)

show_image(image_copy)

#cut out a region of interest (roi)
roi = image[60:160, 320:420] #remember: y, x
show_image(roi)

#resize an image without considering the aspect ratio
resized = cv2.resize(image, (200, 200))
show_image(resized)

#calculate the aspect ratio
# h:w = x:200 --> x = 200*h/w
new_height = 200 * h / w
resized_ratio = cv2.resize(image, (200, int(new_height)))
show_image(resized_ratio)

#rotation of an image
# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
show_image(rotated)

#to mantain all the rotated image into frame we need to calculate the new values 
#of the frame via trigonometry
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
#calculate new frame values getting sine and cosine from the computated matrix
sin = np.abs(M[0, 1])
cos = np.abs(M[0, 0])
nW = int(w * cos + h * sin)
nH = int(w * sin + h * cos)
# adjust the rotation matrix to take into account translation (todo)
M[0, 2] += (nW / 2) - w//2
M[1, 2] += (nH / 2) - h//2
rotated = cv2.warpAffine(image, M, (nW, nH))
show_image(rotated)

#or:
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

#blur an image
# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

#drawing on images: all drawings are made in place so always make a copy of them