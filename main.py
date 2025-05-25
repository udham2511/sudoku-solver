import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import convert_to_tensor
import cv2
import numpy

from keras.models import load_model


def isSudokuValid(board: numpy.ndarray) -> bool:
    """checks if sudoku board is valid or not

    Args:
        board (numpy.ndarray): 9x9 numpy matrix

    Returns:
        bool: valid or not
    """
    for i in numpy.arange(9):
        row = list()
        block = list()
        col = list()

        cuberow = 3 * int(i / 3)
        cubecol = 3 * int(i % 3)

        for j in numpy.arange(9):
            if board[i][j] != 0 and board[i][j] in row:
                return False

            row.append(board[i][j])

            if board[j][i] != 0 and board[j][i] in col:
                return False

            col.append(board[j][i])

            x = cuberow + int(j / 3)
            y = cubecol + int(j % 3)

            if board[x][y] != 0 and board[x][y] in block:
                return False

            block.append(board[x][y])

    return True


def isPositionValid(grid: numpy.ndarray, row: int, col: int, num: int) -> bool:
    """checks if position for placing a specific number in board is valid or not

    Args:
        grid (numpy.ndarray): 9x9 numpy matrix
        row (int): row number where number is to be inserted
        col (int): col number where number is to be inserted
        num (int): number which is to be inserted

    Returns:
        bool: valid move or not
    """
    cubeRow = row - row % 3
    cubeCol = col - col % 3

    for i in numpy.arange(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False

        if grid[i // 3 + cubeRow][i % 3 + cubeCol] == num:
            return False

    return True


def solveSudoku(grid: numpy.ndarray, row: int, col: int) -> bool:
    """solves the sudoku puzzle using recursion

    Args:
        grid (numpy.ndarray): 9x9 numpy matrix
        row (int): row number where number is to be inserted
        col (int): col number where number is to be inserted

    Returns:
        bool: solved or not
    """
    if row == 8 and col == 9:
        return True

    if col == 9:
        col = 0
        row += 1

    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)

    for num in numpy.arange(1, 10):
        if isPositionValid(grid, row, col, num):
            grid[row][col] = num

            if solveSudoku(grid, row, col + 1):
                return True

        grid[row][col] = 0

    return False


def preprocess(image: numpy.ndarray) -> cv2.Mat:
    """apllies different kinds of filters

    Args:
        image (numpy.ndarray): image on which filters are to be applied

    Returns:
        cv2.Mat: processed image
    """
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (9, 9), 0)

    threshedImage = cv2.adaptiveThreshold(
        blurImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2
    )
    morphedImage = cv2.morphologyEx(threshedImage, cv2.MORPH_OPEN, kernel)

    return cv2.dilate(morphedImage, kernel, iterations=1)


def sort(polygon: numpy.ndarray) -> numpy.ndarray:
    """sort corners of rectangle

    Args:
        polygon (numpy.ndarray): array of points of polygon

    Returns:
        numpy.ndarray: sorted points
    """
    polygon = polygon.reshape(4, 2)
    corners = numpy.zeros_like(polygon)

    add = polygon.sum(1)
    corners[0] = polygon[numpy.argmin(add)]
    corners[2] = polygon[numpy.argmax(add)]

    dif = numpy.diff(polygon, axis=1)
    corners[1] = polygon[numpy.argmin(dif)]
    corners[3] = polygon[numpy.argmax(dif)]

    return corners


def makeGridLines(image: numpy.ndarray, index: int, length: int = 10) -> cv2.Mat:
    """make grids on image

    Args:
        image (numpy.ndarray): image on which grids are to be made
        index (int): index
        length (int, optional): width of line. Defaults to 10.

    Returns:
        cv2.Mat: return lines drawn of image
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, image.shape[index] // length)[:: -1 if index else 1]
    )

    erodedImage = cv2.erode(image.copy(), kernel)

    return cv2.dilate(erodedImage, kernel)


def cleanROI(roi: numpy.ndarray) -> cv2.Mat:
    """cleans noise on roi

    Args:
        roi (numpy.ndarray): region of interest

    Returns:
        cv2.Mat: cleaned roi
    """
    if (
        numpy.isclose(roi, 0).sum() / (roi.shape[0] * roi.shape[1]) >= 0.95
        or numpy.isclose(
            roi[
                :,
                int((roi.shape[1] // 2) - roi.shape[1] * 0.4) : int(
                    (roi.shape[1] // 2) + roi.shape[1] * 0.4
                ),
            ],
            0,
        ).sum()
        / (2 * roi.shape[1] * 0.4 * roi.shape[0])
        >= 0.95
    ):
        return blackROI, False

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(
        sorted(contours, key=cv2.contourArea, reverse=True)[0]
    )

    roiCopy = cv2.resize(numpy.zeros_like(roi), (48, 48))

    roiCopy[
        (roiCopy.shape[0] - h) // 2 : (roiCopy.shape[0] - h) // 2 + h,
        (roiCopy.shape[1] - w) // 2 : (roiCopy.shape[1] - w) // 2 + w,
    ] = roi[y : y + h, x : x + w]

    return cv2.resize(roiCopy, (32, 32)), True


kernelGrid = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

blackROI = numpy.zeros((32, 32), numpy.uint8)

model = load_model(r"./model/model.keras")

shape = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, shape[0])
cap.set(4, shape[1])

while cap.isOpened():
    success, frame = cap.read()

    processedFrame = preprocess(frame)

    contours, _ = cv2.findContours(
        processedFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygon = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approx) == 4 and area > 120000:
            polygon = approx
            break

    if polygon is not None:
        corners = sort(polygon)

        length = int(
            max(
                [
                    numpy.linalg.norm(corners[0] - corners[3]),
                    numpy.linalg.norm(corners[0] - corners[1]),
                    numpy.linalg.norm(corners[1] - corners[2]),
                    numpy.linalg.norm(corners[2] - corners[3]),
                ]
            )
        )

        mapping = numpy.array(
            [[0, 0], [length - 1, 0], [length - 1, length - 1], [0, length - 1]],
            dtype=numpy.float32,
        )
        matrix = cv2.getPerspectiveTransform(numpy.float32(corners), mapping)

        wrappedFrame = cv2.warpPerspective(frame, matrix, (length, length))

        processedWrappedFrame = preprocess(wrappedFrame)

        gridWrappedFrame = cv2.add(
            *[makeGridLines(processedWrappedFrame, i) for i in range(2)]
        )
        gridWrappedFrame = cv2.adaptiveThreshold(
            gridWrappedFrame,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            235,
            2,
        )
        gridWrappedFrame = cv2.dilate(gridWrappedFrame, kernelGrid, iterations=2)

        houghLines = numpy.squeeze(
            cv2.HoughLines(gridWrappedFrame, 0.3, numpy.pi / 90, 200)
        )

        for rho, theta in houghLines:
            cos = numpy.cos(theta)
            sin = numpy.sin(theta)

            x1 = int((cos * rho) + 1000 * (-sin))
            y1 = int((sin * rho) + 1000 * cos)
            x2 = int((cos * rho) - 1000 * (-sin))
            y2 = int((sin * rho) - 1000 * cos)

            cv2.line(gridWrappedFrame, (x1, y1), (x2, y2), (255, 255, 255), 5)

        processedWrappedFrame = cv2.bitwise_and(
            processedWrappedFrame, cv2.bitwise_not(gridWrappedFrame)
        )

        digitSquareImage = []
        zerosLocation = set()

        width = wrappedFrame.shape[0] // 9

        for y in numpy.arange(9):
            for x in numpy.arange(9):
                roi = processedWrappedFrame[
                    y * width : (y + 1) * width, x * width : (x + 1) * width
                ]

                roi, isDigit = cleanROI(roi)
                digitSquareImage.append(roi)

                if not isDigit:
                    zerosLocation.add(y * 9 + x % 9)

        predictions = list(
            map(numpy.argmax, model(convert_to_tensor(digitSquareImage)))
        )

        for num, prediction in enumerate(predictions):
            predictions[num] = 0 if num in zerosLocation else prediction + 1

        sudokumatrix = numpy.array(predictions).reshape(9, 9)

        if isSudokuValid(sudokumatrix):
            sudokumatrixCopy = sudokumatrix.copy()

            if solveSudoku(sudokumatrixCopy, 0, 0):
                for y in numpy.arange(9):
                    for x in numpy.arange(9):
                        if y * 9 + x % 9 in zerosLocation:
                            tlPoint = numpy.array([x * width, y * width])
                            brPoint = tlPoint + width

                            center = (tlPoint + brPoint) // 2

                            (tw, th), _ = cv2.getTextSize(
                                str(sudokumatrixCopy[y][x]),
                                cv2.FONT_HERSHEY_COMPLEX,
                                width / 52,
                                2,
                            )

                            cv2.putText(
                                wrappedFrame,
                                str(sudokumatrixCopy[y][x]),
                                (center[0] - tw // 2, center[1] + th // 2),
                                cv2.FONT_HERSHEY_COMPLEX,
                                width / 52,
                                (0, 0, 255),
                                2,
                            )

                matrix, _ = cv2.findHomography(mapping, corners)

                unwrappedFrame = cv2.warpPerspective(
                    wrappedFrame, matrix, frame.shape[:2][::-1]
                )

                cv2.fillConvexPoly(frame, corners, 0)

                frame = cv2.add(frame, unwrappedFrame)

                cv2.drawContours(frame, [polygon], 0, (0, 255, 0), 4)

                for center in corners:
                    cv2.circle(frame, center, 8, (0, 0, 255), -1)

    cv2.imshow("Sudoku Solver", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()

cv2.destroyAllWindows()
