from keras.models import load_model
import cv2 as cv
import numpy as np
from pipeline import Pipeline
#import ntcore as networktables
import os
from os.path import basename
import math
#import argparse
#import logging
from networktables import NetworkTables

#logging.basicConfig(level=logging.DEBUG)


#amongus = "amongus"

########################### NETWORK TABLES #########################

# test server stuff
#parser = argparse.ArgumentParser()
#parser.add_argument("ip", type=str, help="IP address to connect to")
#args = parser.parse_args()

#instance = networktables.NetworkTableInstance.getDefault()

#identity = f"{basename(__file__)}-{os.getpid()}"
#instance.startClient4(identity)
#instance.setServer(server_name=args.ip, port=networktables.NetworkTableInstance.kDefaultPort4)

nt = NetworkTables.getTable('vision')

#instance.setServerTeam(team=3492, port=networktables.NetworkTableInstance.kDefaultPort4)

#table = instance.getTable("vision")
#distance = table.getFloatTopic("distance").publish()
#pixels = table.getFloatTopic("pixels").publish()
#angle = table.getFloatTopic("angle").publish()

###################################################################

np.set_printoptions(suppress=True)

classNames = [0, 1]

height, width = 480, 640
fov = 68.5

camera = cv.VideoCapture(1)

KNOWN_VALUES = [205, 229, 2] # cube width, cone width, distance in ft
CALIBRATE_IMAGES = [cv.imread("images\calibratecone.png"), cv.imread("images\calibratecube.png")]

dualpipecube = Pipeline()
dualpipecone = Pipeline()
dualcalibratecube = Pipeline()
dualcalibratecone = Pipeline()


dualcalibratecone.process(source0=CALIBRATE_IMAGES[0], gametype=0, focalLength=None)
dualcalibratecube.process(source0=CALIBRATE_IMAGES[1], gametype=1, focalLength=None)

focalLengths = []

"""Game piece types: 0 = cone
Game Piece type 1 = cube"""

def calibrateWidthAndFocalLength(gamePieceType: int) -> None:
    if gamePieceType == 0:
        calibratedWidth = float(dualcalibratecone.extract_condata_0_output[4])
    else:
        calibratedWidth = float(dualcalibratecube.extract_condata_1_output[4])

    focalLengths.append((calibratedWidth * KNOWN_VALUES[2]) / KNOWN_VALUES[gamePieceType])

calibrateWidthAndFocalLength(0)
calibrateWidthAndFocalLength(1)

def calculateAngle(differenceInPixels: float) -> float:
    diagonalPixels = math.sqrt(math.pow(height, 2) + math.pow(width, 2))
    degreePerPixel = fov / diagonalPixels
    angle = degreePerPixel * differenceInPixels
    return angle

def findDistanceAndPixels():
    if dualpipecone.find_distance_0_output != None and dualpipecube.find_distance_1_output != None:
        if dualpipecone.find_distance_0_output >= dualpipecube.find_distance_1_output:
            if dualpipecube.extract_condata_1_output != None:
                centerw = dualpipecube.extract_condata_1_output[1]
                difference_in_pix_x = centerw - 320
                nt.putFloat('distance', float(dualpipecone.find_distance_0_output))
                nt.putFloat('pixels', float(difference_in_pix_x))
                nt.putFloat('angle', float(calculateAngle(differenceInPixels=difference_in_pix_x)))

        else:
            if dualpipecone.extract_condata_0_output != None:
                centerw = dualpipecone.extract_condata_0_output[1]
                difference_in_pix_x = centerw - 320
                nt.putFloat('distance', float(dualpipecone.find_distance_0_output))
                nt.putFloat('pixels', float(difference_in_pix_x))
                nt.putFloat('angle', float(calculateAngle(differenceInPixels=difference_in_pix_x)))
    
    if dualpipecube.find_distance_1_output != None and dualpipecone.find_contours_0_output == None:
        if dualpipecube.extract_condata_1_output != None:
            centerw = dualpipecube.extract_condata_1_output[1]
            difference_in_pix_x = centerw - 320
            nt.putFloat('distance', float(dualpipecone.find_distance_0_output))
            nt.putFloat('pixels', float(difference_in_pix_x))
            nt.putFloat('angle', float(calculateAngle(differenceInPixels=difference_in_pix_x)))
    
    if dualpipecone.find_distance_0_output != None and dualpipecube.find_distance_1_output == None:    
        if dualpipecone.extract_condata_0_output != None:
            centerw = dualpipecone.extract_condata_0_output[1]
            difference_in_pix_x = centerw - 320
            nt.putFloat('distance', float(dualpipecone.find_distance_0_output))
            nt.putFloat('pixels', float(difference_in_pix_x))
            nt.putFloat('angle', float(calculateAngle(differenceInPixels=difference_in_pix_x)))

while True:
    ret, image = camera.read()

    
    dualpipecone.process(source0=image, gametype=0, focalLength=focalLengths[0])
    dualpipecube.process(source0=image, gametype=1, focalLength=focalLengths[1])

    findDistanceAndPixels()

    keyboardInput = cv.waitKey(1)
    # 27 = ascii code for escape.
    if keyboardInput == 27:
        break

camera.release()
cv.destroyAllWindows()