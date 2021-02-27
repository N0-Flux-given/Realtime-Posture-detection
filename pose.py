import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import urllib.request

# https://github.com/quanhua92/human-pose-estimation-opencv
# https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py

figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
inWidth = 368
inHeight = 368
thr = 0.2
url = "http://192.168.0.100:8080/shot.jpg"

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], [
                  "RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
                  "Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# All these points represent the head from the side view
HEAD_POINTS = [0, 14, 15, 16, 17]
CHEST_POINTS = [1, 2, 5]
HIP_POINTS = [11, 8]
KNEE_POINTS = [12, 9]
ANKLE_POINTS = [13, 10]

img = cv.imread("woman_side.jpg")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

plt.show()
figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')


def get_pose(frame):
    head_average = None
    chest_average = None
    hip_average = None
    knee_average = None
    ankle_average = None

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                      (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    out = out[:, :19, :, :]

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        point = (int(x), int(y)) if conf > thr else None
        points.append(point)
        # cv.putText(img, list(BODY_PARTS.keys())[list(BODY_PARTS.values()).index(i)], (int(x), int(y)),
        #  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # my stff
        if point is not None and i in HEAD_POINTS:
            if head_average is None:
                head_average = point
            else:
                head_average = (int((point[0] + head_average[0]) / 2),
                                int((point[1] + head_average[1]) / 2))  # Keep updating the average location of the head
        if point is not None and i in CHEST_POINTS:
            if chest_average is None:
                chest_average = point
            else:
                chest_average = (
                    int((point[0] + chest_average[0]) / 2), int((point[1] + chest_average[1]) / 2))
        if point is not None and i in HIP_POINTS:
            if hip_average is None:
                hip_average = point
            else:
                hip_average = (
                    int((point[0] + hip_average[0]) / 2), int((point[1] + hip_average[1]) / 2))
        if point is not None and i in KNEE_POINTS:
            if knee_average is None:
                knee_average = point
            else:
                knee_average = (
                    int((point[0] + knee_average[0]) / 2), int((point[1] + knee_average[1]) / 2))
        if point is not None and i in ANKLE_POINTS:
            if ankle_average is None:
                ankle_average = point
            else:
                ankle_average = (
                    int((point[0] + ankle_average[0]) / 2), int((point[1] + ankle_average[1]) / 2))

    draw(frame, head_average, chest_average,
         hip_average, knee_average, ankle_average)

    print("Head avg ", head_average)
    print("Chest avg ", chest_average)
    print("Hip avg ", hip_average)
    print("Knee avg ", knee_average)
    print("Ankel avg ", ankle_average)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


def draw(frame, head, chest, hip, knee, ankle):
    # Draw lines first
    if head is not None and chest is not None:
        cv.line(frame, head, chest, (0, 255, 0), 3)
        cv.ellipse(frame, head, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    if chest is not None and hip is not None:
        cv.line(frame, chest, hip, (0, 255, 0), 3)
        cv.ellipse(frame, chest, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    if hip is not None and knee is not None:
        cv.line(frame, hip, knee, (0, 255, 0), 3)
        cv.ellipse(frame, hip, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    if knee is not None and ankle is not None:
        cv.line(frame, knee, ankle, (0, 255, 0), 3)
        cv.ellipse(frame, knee, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(frame, ankle, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    if head is not None and chest is not None and hip is not None:
        angle = getAngle(head, chest, hip)
        cv.putText(img, str(angle), chest,
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if chest is not None and hip is not None and knee is not None:
        angle = getAngle(chest, hip, knee)
        cv.putText(img, str(angle), hip,
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


while True:
    img = urllib.request.urlopen(url)
    img_array = np.array(bytearray(img.read()), dtype=np.uint8)
    img = cv.imdecode(img_array, -1)
    pose_img = get_pose(img)
    # plt.imshow(cv.cvtColor(pose_img, cv.COLOR_BGR2RGB))
    cv.imshow("stream", img)
    cv.waitKey(1)
    # plt.show()
