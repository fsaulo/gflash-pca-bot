import time, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Controller
from modules import screenshot
from queue import Queue
from threading import Thread

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

def adjust_to_plot(X, i, dim, orientation='horizontal'):
    M = []
    if orientation == 'horizontal':
        for j in range(i):
            M.append(X[j:j+1,:].reshape(dim[0], dim[1]))
    elif orientation == 'vertical':
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

def play_note(q):
    while True:
        note = q.get()
        keyboard.press(note)
        time.sleep(0.012)
        keyboard.release(note)
        q.task_done()

def sensor(notes):
    pressed = False

    for detection in notes:
        x, y, h, w = detection
        centroid = (x + int(w/2), y + int(h/2))

        if centroid[1] >= 435 and centroid[1] <= 445:
            pressed = True
            print(38*'-' + '\nDETECTED: ({}, {})'.format(centroid[0], centroid[1]))
            if centroid[0] >= 0 and centroid[0] <= 85:
                print('GREEN\n' + 38*'-')
                q1.put('a')
                return pressed, centroid
            elif centroid[0] >= 90 and centroid[0] <= 155:
                print('RED\n' + 38*'-')
                q1.put('s')
                return pressed, centroid
            elif centroid[0] >= 160 and centroid[0] <= 225:
                print('YELLOW\n' + 38*'-')
                q1.put('j')
                return pressed, centroid
            elif centroid[0] >= 230 and centroid[0] <= 295:
                print('BLUE\n' + 38*'-')
                q1.put('k')
                return pressed, centroid
            elif centroid[0] >= 300 and centroid[0] <= 370:
                print('ORANGE\n' + 38*'-')
                q1.put('l')
                return pressed, centroid

    return pressed, (0,0)


if __name__ == '__main__':

    keyboard = Controller()
    nq = 1
    q1 = Queue()
    worker = Thread(target=play_note, args=(q1,))
    worker.setDaemon(True)
    worker.start()
    q1.join()

    # size of the template
    w, h = (50, 60)

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

    lamb, V = PCA(A.T, k=20)
    U = mean(V.T.dot(A))
    R = mean(A)

    # coords of the top left corner
    # screen resolution of the area to be recorded
    # width and height of the processed frame
    x0, y0 = (40, 300)
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
    maxx, maxy = (17, 26)
    threshold = 11.15
    grouping = 0.62
    factor = 1
    pr = 1
    dim = (int(W * (pr)), int(H * pr))

    print('Starting classification...')
    start = time.time(); l = 0; avgt = 0.0
    detected = 0

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

                L = mean(U * (im.reshape(w*h) - R))
                dist = euclidian_dist(U, L)
                d[k] = dist; k += 1

                if dist < threshold:
                    locations.append([dy[0]+5, dx[0]+5, int(w*0.7), int(h*0.78)])
                    locations.append([dy[0]+5, dx[0]+5, int(w*0.7), int(h*0.78)])

            rects, _ = cv2.groupRectangles(locations, factor, grouping)
            for (x, y, w1, h1) in rects:
                top_left = (x, y)
                bottom_right = (x + h1, y + w1)
                cv2.rectangle(img_pad, top_left, bottom_right, (255,255,255), 2)

        # pos, center = sensor(rects)
        #
        # if pos:
        #     cv2.circle(img_pad, center, 15, (255,255,255), thickness=3)
        #     detected += 1

        avgt += (time.time() - start)
        out = cv2.resize(img_pad, dim, interpolation = cv2.INTER_AREA)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print(38*'-' + '\nFrame time = {:.2f}s; average FPS = {:.2f}'.format(avgt/l, l/(avgt)))
            print('Dist. = {:.2f}'.format(d.mean()))
            print('{} notes detected'.format(detected))
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
