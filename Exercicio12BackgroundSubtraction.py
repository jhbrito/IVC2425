# video from http://backgroundmodelschallenge.eu/#evaluation
import cv2
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="Missing --input: path to video",
                    default="vtest.avi")
parser.add_argument("--algo",
                    type=str,
                    help="Missing --algo: Background Subtraction method (MOG2/KNN)",
                    default="MOG2")
args = parser.parse_args()

if args.algo == "MOG2":
    bgsub = cv2.createBackgroundSubtractorMOG2()
    N = bgsub.getNMixtures()
    print("using {} Gaussians".format(N))
    # bgsub.setNMixtures(3)
    # N = bgsub.getNMixtures()
    # print("using {} Gaussians".format(N))
elif args.algo == "KNN":
    bgsub = cv2.createBackgroundSubtractorKNN()
    K = bgsub.getkNNSamples()
    print("using K={} Clusters".format(K))
    # bgsub.setkNNSamples(3)
    # K = bgsub.getkNNSamples()
    # print("using K={} Clusters".format(K))

args.input = os.path.join("Files", args.input)
cap = cv2.VideoCapture(args.input)

# kernel = np.ones((3, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    foreground_mask = bgsub.apply(frame)
    # foreground_mask = cv2.morphologyEx(src=foreground_mask,
    #                          op=cv2.MORPH_OPEN,
    #                          kernel=kernel)
    # foreground_mask = cv2.morphologyEx(src=foreground_mask,
    #                           op=cv2.MORPH_CLOSE,
    #                           kernel=kernel)
    cv2.rectangle(frame,
                  (10, 2),
                  (50, 20),
                  (255, 255, 255),
                  -1)
    cv2.putText(frame,
                str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))),
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0))
    cv2.imshow("frame", frame)

    cv2.imshow("foreground_mask", foreground_mask)
    key = cv2.waitKey(50)
    if key == 27:
        break
