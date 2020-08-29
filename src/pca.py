import time
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ps
import cv2

STAGE = '../images/gf_stage.png'
NOTE1 = '../images/gf_note_blue.png'
NOTE2 = '../images/gf_note_red.png'
NOTE3 = '../images/gf_note_orange.png'
NOTE4 = '../images/gf_note_yellow.png'

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

if __name__ == '__main__':

    set_plt_params(style='dark_background', figdpi=200)

    w, h = (60, 80)
    # img = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE) / 255
    notes = [NOTE1, NOTE2, NOTE3, NOTE4]
    A = np.zeros([len(notes), w*h])

    for index, note in enumerate(notes):
        imgc = cv2.imread(note, cv2.IMREAD_GRAYSCALE) / 255
        A[index:w*h] = imgc.reshape(w*h)

    lamb, V = PCA(A.T, k=2)
    U = mean(V.T.dot(A))

    W, H = (560, 840)
    # img_pad = np.zeros((W, H))
    # img_pad[:img.shape[0],:img.shape[1]] = img
    j = 0; leng = int(W*H/(w*h))

    res = (820, 580)
    x0, y0 = (40, 300)

    t1 = [330,230]
    t2 = [490,230]
    t3 = [205,530]
    t4 = [615,530]

    pts1 = np.float32([t1, t2, t3, t4])
    pts2 = np.float32([[0,0], [W,0], [0,H], [W,H]])

    stepx, stepy = (15, 20)
    maxx = int(W/stepx) + 12
    maxy = int(H/stepy) - 17

    start = time.time()

    while True:
        locations = []
        start = time.time()
        img = ps.grab(bbox=(x0, y0, res[0] + x0, res[1] + y0), backend="maim")
        fps = 1/(time.time() - start)
        d = np.zeros(maxx*maxy); k = 0
        print('FPS = {:.2f}'.format(fps))

        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_pad = cv2.warpPerspective(frame, M, (W,H))

        for i in range(maxx):
            for j in range(maxy):
                dx = (stepx*i, w + (stepx*i))
                dy = (stepy*j, h + (stepy*j))
                im = img_pad[dx[0]:dx[1], dy[0]:dy[1]]

                L = mean(U * (im.reshape(w*h) - mean(A)))
                dist = euclidian_dist(U, L)

                d[k] = dist; k += 1
                threshold = 43.68

                if dist < threshold:
                    locations.append([dy[0], dx[0], int(w*0.7), int(h*0.7)])
                    locations.append([dy[0], dx[0], int(w*0.7), int(h*0.7)])

        rects, _ = cv2.groupRectangles(locations, 1, 0.5)

        for (x, y, w1, h1) in rects:
            top_left = (x, y)
            bottom_right = (x + h1, y + w1)
            cv2.rectangle(img_pad, top_left, bottom_right, (255,255,255), 3)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break

        factor = 0.6
        dim = (int(W * (factor+0.1)), int(H * factor))
        out = cv2.resize(img_pad, dim, interpolation = cv2.INTER_AREA)

        title = 'Screen Capture (Guitar Flash 3) {}x{}'.format(dim[0], dim[1])
        cv2.imshow(title, out)

    # print('Elapsed time for 1 image = {:.2f}s'.format(time.time() - start))
    # print('FPS = {:.1f}'.format(1/(time.time() - start)))
    # print('Found {} points'.format(len(d[d < threshold])))
    # print('Grouped {} rectangles'.format(len(rects)))
