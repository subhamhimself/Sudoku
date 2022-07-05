import keras
import numpy as np
import cv2


def solve(bo):
    # print(bo)
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False


def valid(bo, num, pos):
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i, j) != pos:
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
                return (i, j)
    return None


def srt(x):
    x = x.reshape((4, 2))
    y = np.zeros((4, 1, 2), dtype=np.int32)
    a = x.sum(1)
    b = np.diff(x, axis=1)
    y[0] = x[np.argmin(a)]
    y[3] = x[np.argmax(a)]
    y[1] = x[np.argmin(b)]
    y[2] = x[np.argmax(b)]
    return y


def biggestContour(contours):
    edges = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                edges = approx
                max_area = area
    return edges, max_area


def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = np.array(box)
            box = box[12:40, 12:40]
            boxes.append(box.reshape(784,))
    return np.array(boxes)

model = keras.models.load_model('rough_work/saved.h5')
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
image = 'image.png'
print('\n\n\n\n')
print("Read image")
size = 900
img = cv2.imread(image)
img = cv2.resize(img, (size, size))
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.adaptiveThreshold(img2, 255, 1, 1, 11, 2)

contours = cv2.findContours(img2, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
edges, maxArea = biggestContour(contours)
if edges.size != 0:
    edges = srt(edges)
    cv2.drawContours(img, edges, -1, (0, 0, 255), 25)
    pts1 = np.float32(edges)
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows = np.vsplit(img, 9)
boxes = []
for r in rows:
    cols = np.hsplit(r, 9)
    for box in cols:
        boxes.append(box)
print("Successfully split images")


numbers = []
for image in boxes:
    img = np.asarray(image)
    img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
    img = cv2.resize(img, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predictions = model.predict(img, verbose=0)
    if np.amax(predictions) > 0.7:
        numbers.append(np.argmax(predictions))
    else:
        numbers.append(0)


numbers = np.asarray(numbers)
board = np.array_split(numbers, 9)
print("Initial board :")
print_board(board)
print("\n\nAttempting to solve ")
if(solve(board)):
    print("Solved successfully !\n")
    print_board(board)
else:
    print("Failed to solve")
