import numpy as np
import pyscreenshot as ps
import time
import cv2

SCREEN_SIZE = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("output.avi", fourcc, 15, (SCREEN_SIZE))

res = (820, 580)
x0, y0 = (40, 300)

t1 = [330,230]; t3 = [205,530]
t2 = [490,230]; t4 = [615,530]

pts1 = np.float32([t1, t2, t3, t4])
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])

while True:
    start = time.time()
    img = ps.grab(bbox=(x0, y0, res[0] + x0, res[1] + y0), backend="maim")

    fps = 1/(time.time() - start)
    print('FPS = {:.2f}'.format(fps), end='\r')

    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dsp = cv2.warpPerspective(frame, M, (300,300))

    # video.write(frame)

    factor = 1
    dim = (int(res[0] * factor), int(res[1] * factor))
    out = cv2.resize(dsp, dim, interpolation = cv2.INTER_AREA)

    title = 'Screen Capture (Guitar Flash 3) {}x{}'.format(dim[0], dim[1])
    cv2.imshow(title, dsp)

    if cv2.waitKey(1) == ord("q"): break
