# import the necessary packages
import imutils
import cv2
import numpy as np

def show_image(image, name="Image"):
    # display the image to our screen -- we will need to click the window
    # open by OpenCV and press a key on our keyboard to continue execution
    cv2.imshow(name, image)
    cv2.waitKey(0)

def param_test(image):
    (h, w, d) = image.shape
    print("width={}, height={}, depth={}".format(w, h, d))
    #get RGB values of a pixel
    (B, G, R) = image[100, 50] #x 50, y 100
    print("R={}, G={}, B={}".format(R, G, B))
    show_image(image)
    return (h, w, d)

def remove_red_channel(image):
    h, w, d = image.shape
    #change the image so that the red channel is 0
    image_copy = np.array(image, copy=True)
    for hi in range(h):
        for wi in range(w):
            (B, G, R) = image[hi, wi]
            image_copy[hi, wi] = (B, G, 0)

    show_image(image_copy)

def cutout(image):
    #cut out a region of interest (roi)
    roi = image[60:160, 320:420] #remember: y, x
    show_image(roi)

def resize(image, keep_ratio=False):
    #resize an image without considering the aspect ratio
    h, w, d = image.shape
    if keep_ratio:
        resized = cv2.resize(image, (200, 200))
    else:
        #calculate the aspect ratio
        # h:w = x:200 --> x = 200*h/w
        new_height = 200 * h / w
        resized = cv2.resize(image, (200, int(new_height)))
    show_image(resized)

def rotate(image, bounded=False):
    #rotation of an image
    # let's rotate an image 45 degrees clockwise using OpenCV by first
    # computing the image center, then constructing the rotation matrix,
    # and then finally applying the affine warp
    h, w, d = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    if not bounded:
        nW = w
        nH = h
    else:
        #to mantain all the rotated image into frame we need to calculate the new values 
        #of the frame via trigonometry
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

def rotate_ez(image):
    rotated = imutils.rotate_bound(image, 45)
    show_image(rotated)

def blur(image):
    #blur an image
    # apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
    # useful when reducing high frequency noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    show_image(blurred)

def draw(image):
    #drawing on images: all drawings are made in place so always make a copy of them
    output = image.copy()
    #precalculated rectangle on image with color red and thickness 2
    cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
    cv2.circle(output, (300, 150), 20, (255, 0, 0), -1) #-1 thickness fills the figure
    cv2.line(output, (60, 20), (400, 200), (0, 255, 0), 5)
    cv2.putText(output, "OpenCV + Jurassic Park!!!", (250, 40), 
	                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Rectangle", output)
    cv2.waitKey(0)

def first_part_showcase():
    image = cv2.imread("jp.png")
    param_test(image)
    remove_red_channel(image)
    cutout(image)
    resize(image)
    resize(image, keep_ratio=True)
    rotate(image)
    rotate(image, bounded=True)
    rotate_ez(image)
    blur(image)
    draw(image)

def second_part_showcase():
    pass

def main():
    #showcase some of the experiments
    first_part_showcase()
    second_part_showcase()

if __name__ == "__main__":
    main()

