import numpy as np
import cv2

CHECKERBOARD = (6, 9)
CRITERIA = (cv2.TERM_CRITERA_EPIS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
FLAGS = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
imageShape = None
objPoints = []
imagePoints = []

for fname in images:
    image = cv2.imread(fname)
    if imageShape == None:
        imageShape = image.shape[:2]
    else:
        assert imageShape == image.shape[:2], "Images must share their size."
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardConers(grayscale, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objPoints.append(objp)
        cv2.cornerSubPix(grayscale, corners, (3, 3), (-1, -1), CRITERIA)
        imagePoints.append(corners)

objPointsLen = len(objPoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(objPointsLen)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(objPointsLen)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objPoints,
        imagePoints,
        grayscale.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        FLAGS,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

dim = imageShape[::-1]
balance = 1
dim2, dim3 = None, None

image = cv2.imread("left12.jpg")
dim1 = image.shape[:2][::-1]
assert dim1[0]/dim1[1] == dim[0]/dim[1], "Image to undistort needs to have same aspect ratio"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaledK = K * dim1[0] / dim[0]
scaledK[2][2] = 1.0

newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaledK, D, dim2, np.eye(3), balance=balance)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaledK, D, np.eye(3), newK, dim3, cv2.CV_16SC2)

undistortedImage = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

data = {'dim1': dim1, 
        'dim2':dim2,
        'dim3': dim3,
        'K': np.asarray(K).tolist(), 
        'D':np.asarray(D).tolist(),
        'new_K':np.asarray(newK).tolist(),
        'scaled_K':np.asarray(scaledK).tolist(),
        'balance':balance}

import json
with open("fisheye_calibration_data.json", "w") as f:
    json.dump(data, f)

cv2.imshow("undistorted", undistortedImage)
