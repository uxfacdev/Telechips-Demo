import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

model = tf.keras.models.load_model(
    './testweight-30', custom_objects=None, compile=True
)

import cv2
import numpy as np


def makeFrame(frame, w, h, text, value, fn, fe):
    center_x = int(w / 2.0)
    center_y = int(h / 2.0)

    thickness = 2

    location = (10, 20)
    # font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    yellow = (0, 255, 255)
    cv2.putText(frame, text, location, font, fontScale, yellow, thickness)

    outLocation =  (100, 100)
    outFont = cv2.FONT_HERSHEY_SIMPLEX
    outFontScale = 1
    outColor = (0, 255, 0)

    cv2.putText(frame, value, outLocation, outFont, outFontScale, outColor, thickness)

    frameLocation =  (10, 300)
    frameFont = cv2.FONT_HERSHEY_SIMPLEX
    frameFontScale = 2
    frameColor = (0, 0, 255)

    frameText = str(fe) + " / " + str(fn) + " " + str('%0.2f' % float(((fn-fe)/fn) * 100)) + "%"

    cv2.putText(frame, frameText, frameLocation, frameFont, frameFontScale, frameColor, thickness)

    return frame


def useVideotoModel(vp, model):
    cap = cv2.VideoCapture(vp)

    maxPixel = 255.0
    width = cap.get(3)  # float
    height = cap.get(4)  # float

    frameNumber = 0
    frameMap = {}
    frameError = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break
        frameNumber += 1


        image = cv2.resize(src=frame, dsize=(16, 16), interpolation=cv2.INTER_AREA)
        imageSet = []
        imageSet.append(image / maxPixel)
        imageSet = np.array(imageSet)

        tensor = model(imageSet)
        result = str(tensor)

        # print(tf.math.argmax(tensor, 1))
        # print(tf.math.argmax(tensor, 1).numpy())
        outValue = tf.math.argmax(tensor, 1).numpy()

        if frameMap.get(outValue[0]) == None:
            frameMap[outValue[0]] = 1
        else:
            frameMap[outValue[0]] += 1
        outTextValue = ""
        if outValue == 0:
            outTextValue = "tin"
        elif outValue == 1:
            outTextValue = "tinner"
        elif outValue == 2:
            outTextValue = "tout"
        elif outValue == 3:
            outTextValue = "touter"

        if frameNumber % 5 == 0:
            print(frameMap)
            if len(frameMap.keys()) > 1:
                frameError += 1
            frameMap = {}

        frame = makeFrame(frame, width, height, result, outTextValue, frameNumber, frameError)
        cv2.imshow('test', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("break")
            break



    cap.release()

    print(frameMap)


# useVideotoModel('testVideo.mp4', model)
# useVideotoModel('../test_tunnel_1.mp4', model)
useVideotoModel('../datasets-video/hak_tunnel040.mp4', model)