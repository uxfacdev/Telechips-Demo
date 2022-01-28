import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

model = tf.keras.models.load_model(
    './testweight', custom_objects=None, compile=True
)

import cv2
import numpy as np


def makeFrame(frame, w, h, text):
    center_x = int(w / 2.0)
    center_y = int(h / 2.0)

    thickness = 2

    location = (center_x - 200, center_y - 100)
    location = (10, 20)
    # font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    yellow = (0, 255, 255)
    cv2.putText(frame, text, location, font, fontScale, yellow, thickness)

    return frame


def useVideotoModel(vp, model):
    cap = cv2.VideoCapture(vp)

    width = cap.get(3)  # float
    height = cap.get(4)  # float

    maxPixel = 255.0

    while (cap.isOpened()):
        ret, frame = cap.read()

        image_resize = cv2.resize(src=frame, dsize=(16, 16), interpolation=cv2.INTER_AREA)
        image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        image_resize = cv2.merge((image_resize, image_resize, image_resize))
        imageSet = []
        imageSet.append(image_resize / maxPixel)
        imageSet = np.array(imageSet)

        tensor = model(imageSet)
        result = str(tensor)

        # print(tensor.shape, tensor.__iter__())
        # print(result)
        # import sys
        # tf.print(tensor, output_stream=sys.stderr)

        print(tf.math.argmax(tensor, 1))
        frame = makeFrame(frame, width, height, result)
        # cv2.imshow('test', frame)

        cv2.imshow('test', image_resize)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("break")
            break

    cap.release()


# useVideotoModel('testVideo_1.mp4', model)
useVideotoModel('../test_tunnel_1.mp4', model)
