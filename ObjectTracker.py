# import the necessary packages
from CentroidTracking.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True,
	help="path to input video")
# ap.add_argument("-o", "--output", required=True,
	# help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

ct = CentroidTracker()
# load the COCO class labels, our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

writer = None

if args["input"] == 'camera':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args["input"])

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


print(cap.isOpened())
print("starting-----------------------------------------------------------")
begin = time.time()
while (cap.isOpened()):
    ret, image = cap.read()

    # load our input image and grab its spatial dimension

    if ret == True:
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        	swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        boxes_c = []
        confidences = []
        classIDs = []
        rects = []
        # loop over each of the layer outputs
        for output in layerOutputs:
        	# loop over each of the detections
            for detection in output:
        		# extract the class ID and confidence (i.e., probability) of
        		# the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
        			# use the center (x, y)-coordinates to derive the top and
        			# and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
        			# update our list of bounding box coordinates, confidences,
        			# and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    boxes_c.append([centerX - int(width/2), centerY - int(height/2), centerX + int(width/2), centerY + int(height/2)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                rects.append(boxes_c[i])
        # update our centroid tracker using the computed set of bounding
    	# box rectangles
        objects = ct.update(rects)
    	# loop over the tracked objects
        for (objectID, centroid) in objects.items():
    		# draw both the ID of the object and the centroid of the
    		# object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # ensure at least one detection exists
        if len(idxs) > 0:
        	# loop over the indexes we are keeping
        	for i in idxs.flatten():
        		# extract the bounding box coordinates
        		(x, y) = (boxes[i][0], boxes[i][1])
        		(w, h) = (boxes[i][2], boxes[i][3])

        		# draw a bounding box rectangle and label on the image
        		color = [int(c) for c in COLORS[classIDs[i]]]
        		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        			0.5, color, 2)
        # writer.write(image)
        # # show the output image
        # cv2.imshow("Image", image)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30,(image.shape[1], image.shape[0]), True)
            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
        cv2.imshow("Live", image)
        # write the output frame to disk
        writer.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
cap.release()
cv2.destroyAllWindows()
finish = time.time()

print(f"Total time taken : {finish - begin}")
