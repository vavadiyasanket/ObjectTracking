# Object Tracking
### Object tracking is the process of:
1. Taking an initial set of object detections
2. Creating a unique ID for each of the initial detections
3. And then tracking each of the objects as they move around frames in a video, maintaining the assignment of unique IDs  
As described in first step, we should detect the from the each frame of video. Here *YOLO object detection* is used to detect the object in image/video frame. Implemention of yolo is in ```ObjectTracker.py```. Next two steps are implemented in ```CentroidTracking/centroidtracker.py```.

To run on video file:
  ```python ObjectTracker.py --input videos/car_chase_01.mp4 --confidence 0.5 --threshold 0.3```  
To run on web-camera:
  ```python ObjectTracker.py --input camera --confidence 0.5 --threshold 0.3```  
  
references/credit:  
<a href="https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/" terget="_blank">YOLO object detection with OpenCV</a>  
<a href="https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/" terget="_blank">Simple object tracking with OpenCV</a>
