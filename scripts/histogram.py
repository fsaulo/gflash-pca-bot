import matplotlib.pyplot as plt
import cv2

stage = cv2.imread('GF_transf_stage_3.png', cv2.IMREAD_GRAYSCALE)

plt.hist(stage.flatten(), 256,[0,256])
plt.show()
