import datetime

import numpy as np
import cv2
from utils import detector

detection_graph, sess = detector.load_inference_graph()
#start the video
start_time = datetime.datetime.now()
num_frames = 0
num_of_vehicle=100
threshold=0.40
Line_Perc2=float(30)
cap = cv2.VideoCapture("traffic.mp4")
while (True):
    ret,frame = cap.read()
    frame = np.array(frame)
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
    # cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

    # cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

    # Run image through tensorflow graph
    num_frames += 1
    im_height, im_width = frame.shape[:2]
    print(im_width)
    print(im_height)
    boxes, scores, classes = detector.detect_objects(
        frame, detection_graph, sess)
    print(len(scores))
    print(len(classes))

    a, b = detector.draw_box_on_image(
        num_of_vehicle, threshold, scores, boxes, classes, im_width, im_height, frame)

    #(left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                  #boxes[i][0] * im_height, boxes[i][2] * im_height)
    #cv2.rectangle(image_np, p1, p2, color , 3, 1)
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time
    print('*****************************************************************')
    print('FPS==========================',fps)
    print('*****************************************************************')
    """
    your code here
    """
    cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()