import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ps
import itertools
from modules import screenshot

def PCA(M, k=2):
    x1, x2 = M.shape
    if (x1 >= x2): M = M.T
    R = np.cov(M - mean(M))
    lamb, Vy = np.linalg.eigh(R)
    return lamb, Vy[:,:k]

def imshow(A, grid=True):
    fig, axs = plt.subplots(1, len(A))
    for ax, im in zip(axs, A):
        ax.imshow(im, cmap=plt.cm.gray_r)
        ax.grid(grid)

def linfit(set):
    M = np.matrix([np.ones(len(set)), range(len(set))]).T
    theta = (np.linalg.inv(M.T * M) * M.T).dot(set)
    return M*theta.T

def euclidian_dist(V, Vt=None):
    return np.linalg.norm(V - Vt)

def set_plt_params(figsize=[5,5], figdpi=150, style='default'):
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = figdpi
    plt.style.use(style)

def adjust_to_plot(X, i, dim):
    x1, x2 = X.shape
    M = []
    if x1 == i:
        for j in range(i):
            M.append(X[j:j+1,:].reshape(dim[0], dim[1]))
    else:
        for j in range(i):
            M.append(X[:,j:j+1].reshape(dim[0], dim[1]))
    return M

def normalize(M):
    return M/np.linalg.norm(M)

def mean(M):
    ux = np.zeros(M[:1,].shape)
    for m in M:
        ux += m
    return ux/len(M)

def cut_img(U, pos, size=(60,80)):
    return U[pos[0]:size[0]+pos[0],pos[1]:size[1]+pos[1]]

def pre_processing():
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255


def perspective_transform(frame, x1, x2, width, height):
    M = cv2.getPerspectiveTransform(x1, x2)
    return cv2.warpPerspective(frame, M, (width, height))


if __name__ == '__main__':

    set_plt_params(style='dark_background', figdpi=200)

    # size of the template
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

    # coords of the top left corner
    # screen resolution of the area to be recorded
    # width and height of the processed frame
    x0, y0 = (40, 400)
    res = (820, 580)
    W, H = (380, 550)

    # position of the inner stage, where the notes are placed
    t1 = [330,230]
    t2 = [490,230]
    t3 = [205,530]
    t4 = [615,530]

    pts1 = np.float32([t1, t2, t3, t4])
    pts2 = np.float32([[0,0], [W,0], [0,H], [W,H]])

    # this is the frame search, if stepx, stepy < w, h the exceeding sizes
    # will overlap the search window making the classification more accurate
    # in exchange for computing power
    stepx, stepy = (20, 20)
    maxx, maxy = (17, 24)
    threshold = 28.8
    grouping = 0.56
    factor = 1
    pr = 1
    dim = (int(W * (pr+0.1)), int(H * pr))

    print('Starting classification...')
    start = time.time(); l = 0; avgt = 0.0

    while True:
        locations = []
        d = np.zeros(maxx*maxy); k = 0
        start = time.time(); l += 1

        img = screenshot.grab_screen(x0, y0, res[0] + x0, res[1] + y0)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 256
        img_pad = perspective_transform(frame, pts1, pts2, W, H)
        img_screenshot = img_pad.copy()

        for i in range(maxy):
            for j in range(maxx):
                dx = (stepx*i, w + (stepx*i))
                dy = (stepy*j, h + (stepy*j))

                im = img_pad[dx[0]:dx[1], dy[0]:dy[1]]

                L = mean(U * (im.reshape(w*h) - mean(A)))
                dist = euclidian_dist(U, L)
                d[k] = dist; k += 1

                if dist < threshold:
                    locations.append([dy[0]+5, dx[0]+5, int(w*0.75), int(h*0.7)])
                    locations.append([dy[0]+5, dx[0]+5, int(w*0.75), int(h*0.7)])

            rects, _ = cv2.groupRectangles(locations, factor, grouping)
            for (x, y, w1, h1) in rects:
                top_left = (x, y)
                bottom_right = (x + h1, y + w1)
                cv2.rectangle(img_pad, top_left, bottom_right, (255,255,255), 2)

        avgt += (time.time() - start)
        out = cv2.resize(img_pad, dim, interpolation = cv2.INTER_AREA)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print(38*'-' + '\nFrame time = {:.2f}s; average FPS = {:.2f}'.format(avgt/l, l/(avgt)))
            print('Dist. = {:.2f}'.format(d.mean()))
            print('End program.\n'+ 38*'-')
            cv2.destroyAllWindows()
            break

        elif key == ord('s'):
            path = '../images/screenshots/screenshot_{}_stage.png'.format(int(time.time()))
            output = np.array(img_screenshot * 256, dtype=np.uint8)
            cv2.imwrite(path, output)
            print('Screenshot taken saved at ' + path)

        title = 'Screen Capture (Guitar Flash 3) {}x{}'.format(dim[0], dim[1])
        cv2.imshow(title, out)
