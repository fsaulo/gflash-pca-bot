import time
import numpy as np
import matplotlib.pyplot as plt
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
    img = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE) / 255
    notes = [NOTE1, NOTE2, NOTE3, NOTE4]
    A = np.zeros([len(notes), w*h])

    for index, note in enumerate(notes):
        imgc = cv2.imread(note, cv2.IMREAD_GRAYSCALE) / 255
        A[index:w*h] = imgc.reshape(w*h)

    lamb, V = PCA(A.T, k=4)
    U = normalize(mean(V.T.dot(A)))

    W, H = (840, 560)
    img_pad = np.zeros((W, H))
    img_pad[:img.shape[0],:img.shape[1]] = img
    j = 0; leng = int(W*H/(w*h))

    stepx, stepy = (20, 30)
    maxx = int(W/stepx) - 10
    maxy = int(H/stepy) - 1

    d = np.zeros(maxx*maxy); k = 0
    locations = []

    start = time.time()

    for i in range(maxx):
        dx = (stepx*i, w + (stepx*i))
        for j in range(maxy):
            dy = (stepy*j, h + (stepy*j))
            im = img_pad[dx[0]:dx[1], dy[0]:dy[1]]

            L = normalize(mean(U * (im.reshape(w*h) - mean(A))))
            dist = euclidian_dist(U, L)

            d[k] = dist; k += 1
            threshold = 1.925

            if dist < threshold:
                locations.append([dy[0], dx[0] - 10, int(w*1.2), int(h)])
                locations.append([dy[0], dx[0], int(w), int(h)])

    rects, _ = cv2.groupRectangles(locations, 1, 0.4)

    print('Elapsed time for 1 image = {:.2f}s'.format(time.time() - start))
    print('FPS = {:.1f}'.format(1/(time.time() - start)))
    print('Found {} points'.format(len(d[d < threshold])))
    print('Grouped {} rectangles'.format(len(rects)))

    for (x, y, w, h) in rects:
        top_left = (x, y)
        bottom_right = (x + h, y + w)
        cv2.rectangle(img, top_left, bottom_right, (255,255,255), 3)

    cv2.imshow('', img)
    cv2.waitKey()
