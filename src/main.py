import os
import numpy as np
import pyscreenshot as ps
import cv2
import time
from modules import screenshot
from pca import *

STAGE = '/home/felix/Documents/Projects/gflash/images/screenshots/screenshot_1598836427.png'
stage = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE) / 256

if __name__ == '__main__':

    w, h = (50, 60);
    print(38*'-' + '\nLoading templates...')
    templates_path = '../images/classification/'
    files = os.listdir(templates_path)
    A = np.zeros([len(files), w * h])

    for index, file in enumerate(files):
        note = os.path.join(templates_path, file)
        imgc = cv2.imread(note, cv2.IMREAD_GRAYSCALE) / 256
        imgc = cv2.resize(imgc, (h, w), interpolation = cv2.INTER_AREA)
        A[index:w*h] = imgc.reshape(w*h)
        print(note + ' loaded!')

    print('{} templates loaded & read for classification'.format(len(A)))
    print(38*'-' + '\nCalculating a base for templates...')

    lamb, V = PCA(A.T, k=5)
    U = mean(V.T.dot(A))

    x0, y0 = (20, 290)
    res = (820, 580)
    W, H = (380, 550)

    t1 = [330,230]
    t2 = [490,230]
    t3 = [205,530]
    t4 = [615,530]

    pts1 = np.float32([t1, t2, t3, t4])
    pts2 = np.float32([[0,0], [W,0], [0,H], [W,H]])

    stepx, stepy = (20, 20)
    maxx, maxy = (17, 24)
    threshold = 23
    grouping = 0.6
    factor = 0
    pr = 1
    dim = (int(W * (pr+0.1)), int(H * pr))

    print('Starting classification...')

    d = np.zeros(maxx*maxy); k = 0
    start = time.time();

    img_pad = perspective_transform(stage, pts1, pts2, W, H)
    img_screenshot = img_pad.copy()
    locations = []

    for i in range(maxy):
        for j in range(maxx):
            dx = (stepx*i, w + (stepx*i))
            dy = (stepy*j, h + (stepy*j))

            im = img_pad[dx[0]:dx[1], dy[0]:dy[1]]

            L = mean(U * (im.reshape(w*h) - mean(A)))
            dist = euclidian_dist(U, L)
            d[k] = dist; k += 1

            if dist < threshold:
                locations.append([dy[0], dx[0], int(w*0.85), int(h*0.8)])
                locations.append([dy[0], dx[0], int(w*0.85), int(h*0.8)])

            rects, _ = cv2.groupRectangles(locations, factor, grouping)
        for (x, y, w1, h1) in rects:
            top_left = (x, y)
            bottom_right = (x + h1, y + w1)
            cv2.rectangle(img_pad, top_left, bottom_right, (255,255,255), 1)

    avgt = (time.time() - start)
    img_pad = np.array(-img_pad * 256, dtype=np.uint8)
    out = cv2.resize(img_pad, dim, interpolation = cv2.INTER_AREA)
    key = cv2.waitKey(0)

    if key == ord('q'):
        print(38*'-' + '\nFrame time = {:.2f}s; average FPS = {:.2f}'.format(avgt, 1/(avgt)))
        print('Dist. = {:.2f}'.format(d.mean()))
        print('End program.\n'+ 38*'-')
        cv2.destroyAllWindows()

    title = 'Screen Capture (Guitar Flash 3) {}x{}'.format(dim[0], dim[1])
    cv2.imshow(title, out)
    # cv2.imwrite('img.png', out)
