import cv2
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="Missing --input: path to video",
                    default="vtest.avi")

args = parser.parse_args()

args.input = os.path.join("Files", args.input)
cap = cv2.VideoCapture(args.input)

ret, frame = cap.read()
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imshow("frame", frame)
    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev=previous_frame,
                                        next=next_frame,
                                        flow=None,
                                        pyr_scale=0.25,
                                        levels=1,
                                        winsize=5,
                                        iterations=1,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    flow_norm = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    flow_norm_normalized = cv2.normalize(flow_norm,
                                         None,
                                         0.0,
                                         1.0,
                                         cv2.NORM_MINMAX)
    cv2.imshow("flow_norm_normalized", flow_norm_normalized)
    key = cv2.waitKey(50)
    if key == 27:
        break
    previous_frame = next_frame

