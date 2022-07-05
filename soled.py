# %%
# from tensorflow.keras.models import load_model
import imp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow as tf
import numpy as np
import cv2
# import image_processing
# from image_processing import *

# image_processing


def solve(bo):
    # print(bo)
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False
def preProcess(img):
    # return img
    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # imgBlur = imgGray
    # return imgBlur
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold

def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True

def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None


def show1(a):
    plt.imshow(a, cmap=matplotlib.cm.binary, interpolation='nearest')
    return plt.axis('off')
def show(a):
    a = a.reshape(28, 28)
    plt.imshow(a, cmap=matplotlib.cm.binary, interpolation='nearest')
    return plt.axis('off')


# 1 - Preprocessing Image
def preProcess(img):
    # return img
    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # imgBlur = imgGray
    # return imgBlur
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


# 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew


# 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = np.array(box)
            box = box[12:40,12:40]
            boxes.append(box.reshape(784,))
    return np.array(boxes)


# 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes, model):
    result = []
    for image in boxes:
        # PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))

        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        # GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        # SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex)
        else:
            result.append(0)
    # model.predict(boxes)
    
    return result


# 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)
                             ), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


# 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range(0, 9):
        pt1 = (0, secH*i)
        pt2 = (img.shape[1], secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img


# 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver


# %%
model = keras.models.load_model('saved.h5')
model.compile(optimizer='adam',run_eagerly=True,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
image = 'ez.png'
# image = 'sample2.png'
# image = 'sample.jpg'
img_size = 1800


# %%
img = cv2.imread(image)
img = cv2.resize(img, (img_size, img_size))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((img_size, img_size, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

# %%
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
# cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
# show1(imgContours)

# %%
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
# print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[img_size, 0], [0, img_size],[img_size, img_size]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (img_size, img_size))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
# stackImages(imgWarpColored)
# show1(imgWarpColored)


# %%
imgSolvedDigits = imgBlank.copy()


def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
# boxes = splitBoxes(imgWarpColored)


# boxes[1].shape
img = imgWarpColored
# cropped = img[200:300, 150:250]
rows = np.vsplit(img, 9)
boxes = []
for r in rows:
    cols = np.hsplit(r, 9)
    for box in cols:
        boxes.append(box)


# rows = np.vsplit(img, 9)
# boxes = []
# for r in rows:
#     cols = np.hsplit(r, 9)
#     for box in cols:
#         box = np.array(box)
#         # box = cv2.resize(box, (100,100), interpolation=cv2.INTER_CUBIC)
#         # box = box[10:85,15:85]
#         # box = 255-box
#         # box = cv2.resize(box, (50, 50), interpolation=cv2.INTER_CUBIC)
#         box -= box*(box<=50)
#         # box += (200-box)*(box>50)*(box<200)
#         # box += (255-box)*(box>50)
#         # boxes.append(box.reshape(784,))

# boxes = np.array(boxes)

# t = model.predict(boxes)
# t = model.predict(boxes)
# t = getPredection(boxes, model)
numbers = getPredection(boxes, model)



# %%

# p = 10

# imgDetectedDigits = displayNumbers(
#     imgDetectedDigits, numbers, color=(255, 0, 255))
numbers = np.asarray(numbers)
# print(numbers)
posArray = np.where(numbers > 0, 0, 1)
# print(posArray)

# 5. FIND SOLUTION OF THE BOARD
board = np.array_split(numbers, 9)
# print(board)
solve(board)
print(board)

# board

# %%


# %%



