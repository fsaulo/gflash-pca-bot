import cv2
import numpy as np

img = cv2.imread('GF_stage_3.png')
rows, cols, _ = img.shape

pts1 = np.float32([[450,330],[680,330],[285,730],[850,730]])
pts2 = np.float32([[0,0], [rows,0], [0,cols], [rows,cols]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (rows, cols))

# img_out = cv2.flip(dst, 0)

cv2.imwrite('GF_transf_stage_3.png', dst)
