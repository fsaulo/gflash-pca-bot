import cv2

reftPt = []
panel_img = cv2.imread('GF_stage_3.png', cv2.IMREAD_GRAYSCALE)

def click(event, x, y, flags, param):
    global reftPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print(refPt)

cv2.imshow('img', panel_img)
cv2.setMouseCallback('img', click)
cv2.waitKey()
