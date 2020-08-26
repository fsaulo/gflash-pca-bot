import numpy as np
import cv2

STAGE = 'gf_stage.png'
NOTE = 'gf_note.png'

stage = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE)
template = cv2.imread(NOTE, cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(stage, template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
w, h = template.shape[::-1]
threshold = 0.65
loc = np.where(result >= threshold)

for pt in zip(*loc[::-1]):
    tr, bl = pt[0] + int(w), pt[1] + int(h)
    white = (255, 255, 255)
    if (pt[1] < 1150): cv2.rectangle(stage, pt, (tr, bl), white, 2)

factor = 0.5
dim = (int(stage.shape[1] * factor), int(stage.shape[0] * factor))
out = cv2.resize(stage, dim, interpolation = cv2.INTER_AREA)

cv2.imwrite('output.png', out)
