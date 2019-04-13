import numpy as np
import cv2
def detectFailure(img):
    kernel = np.array([[0, -1, 0], [-1,6, -1], [0, -1, 0]], np.float32) #锐化
    img = cv2.filter2D(img, -1, kernel=kernel)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,211,21)

    image, contours, hierarchy = cv2.findContours(th1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros(th1.shape, dtype = np.uint8)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if  area > 10000:
            continue
        area = cv2.contourArea(contours[i])
        cv2.drawContours(canvas,contours,i,255,1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    #canvas = cv2.erode(canvas,disc,iterations = 1)
    canvas = cv2.dilate(canvas,disc,iterations = 6)

    image, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    track_window = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if  area < 400 or hierarchy[0][i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        track_window.append((x,y,w,h))

    return track_window

def detectEx(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    canvas = np.zeros(thresh.shape, dtype = np.uint8)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if  hierarchy[0][i][3] != -1 or hierarchy[0][i][2] != -1 :
            continue
        cv2.drawContours(canvas,contours,i,255,1)

    kernel = np.ones((9,9),dtype = np.uint8)
    canvas = cv2.dilate(canvas, kernel,iterations = 9)
    track_window =[]
    image, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        track_window.append((x,y,w,h))
    return track_window