import matplotlib.pyplot as plt
import numpy as np
import cv2

stage = cv2.imread('GF_transf_stage.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('GF_transf_green_note.png', cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(stage, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

w, h = template.shape[::-1]
threshold = 0.6
loc = np.where(result >= threshold)

for pt in zip(*loc[::-1]):
    if (pt[1] < 1150):
        cv2.rectangle(stage, pt, (pt[0] + int(w*1.5), pt[1] + int(h*1.5)), (255,255,255), 2)

dim = (int(stage.shape[1] * 0.75), int(stage.shape[0] * 0.75))
out = cv2.resize(stage, dim, interpolation = cv2.INTER_AREA)

cv2.imshow('Output', out)
cv2.waitKey()

print(result)
