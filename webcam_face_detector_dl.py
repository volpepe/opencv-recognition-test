# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    return args

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
    args = parse_args()

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    
    fps_calc_frames = 5
    rendered_frames = 0
    fps = 0.0

    while True:
        if rendered_frames == 0: 
            start = time.time()
        # grab the frame from the threaded video stream and resize it
	    # to have a maximum width of 400 pixels
        ret, img = cap.read()
        img = resize(img, newDimW=800, keep_ratio=True)

        # grab the frame dimensions and convert it to a blob
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

        if rendered_frames == fps_calc_frames:
            rendered_frames = 0
            end = time.time()
            total_time = end - start
            fps = fps_calc_frames / total_time
        else:
            rendered_frames+=1

        fps_str = "FPS: {:06.2f}".format(fps)
        cv2.putText(img, fps_str, (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()