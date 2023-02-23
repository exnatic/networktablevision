from keras.models import load_model
import cv2 as cv
import numpy as np
from pipeline import Pipeline
import ntcore as networktables
import os
from os.path import basename
#import argparse
#import logging

#logging.basicConfig(level=logging.DEBUG)

# test server stuff
#parser = argparse.ArgumentParser()
#parser.add_argument("ip", type=str, help="IP address to connect to")
#args = parser.parse_args()

instance = networktables.NetworkTableInstance.getDefault()

identity = f"{basename(__file__)}-{os.getpid()}"
instance.startClient4(identity)

instance.setServerTeam(team=3492, port=networktables.NetworkTableInstance.kDefaultPort4)

table = instance.getTable("vision")
distance = table.getFloatTopic("distance").publish()
pixels = table.getFloatTopic("pixels").publish()

np.set_printoptions(suppress=True)

classNames = [0, 1]

camera = cv.VideoCapture(0)

KNOWN_VALUES = [205, 229, 2] # cube width, cone width, distance in ft
CALIBRATE_IMAGES = [cv.imread("images\calibratecone.png"), cv.imread("images\calibratecube.png")]

processedCone = Pipeline()
processedCube = Pipeline()
processedCone.process(source0=CALIBRATE_IMAGES[0], gametype=0, focalLength=None)
processedCube.process(source0=CALIBRATE_IMAGES[1], gametype=1, focalLength=None)

focalLengths = []

"""Game piece types: 0 = cone
Game Piece type 1 = cube"""

def calibrateWidthAndFocalLength(gamePieceType: int) -> None:
    if gamePieceType == 0:
        calibratedWidth = float(processedCone.extract_condata_0_output[4])
    else:
        calibratedWidth = float(processedCube.extract_condata_1_output[4])

    focalLengths.append((calibratedWidth * KNOWN_VALUES[2]) / KNOWN_VALUES[gamePieceType])

calibrateWidthAndFocalLength(0)
calibrateWidthAndFocalLength(1)

def findDistanceAndPixels():
    if processedCone.find_distance_0_output != None \
        and processedCube.find_distance_1_output != None:
            if processedCone.find_distance_0_output >= processedCube.find_distance_1_output:
                if processedCube.extract_condata_1_output != None:
                    centerW = processedCube.extract_condata_1_output[1]
                    differenceInPixels = centerW - 320
                    distance.set(float(processedCube.find_distance_1_output))
                    pixels.set(float(differenceInPixels))
            else:
                if processedCone.extract_condata_0_output != None:
                    centerW = processedCone.extract_condata_0_output[1]
                    differenceInPixels = centerW - 320
                    distance.set(value=float(processedCube.find_distance_0_output), time=0)
                    pixels.set(value=float(differenceInPixels), time=0)

while True:
    ret, image = camera.read()

    processedCone.process(source0=image, gametype=0, focalLength=focalLengths[0])
    processedCube.process(source0=image, gametype=1, focalLength=focalLengths[1])

    findDistanceAndPixels()

    keyboardInput = cv.waitKey(1)
    # 27 = ascii code for escape.
    if keyboardInput == 27:
        break

camera.release()
networktables.NetworkTableInstance.destroy()
cv.destroyAllWindows()