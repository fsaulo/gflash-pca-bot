import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

STAGE = 'gf_stage.png'
NOTE1 = 'gf_note_blue.png'
NOTE2 = 'gf_note_red.png'
NOTE3 = 'gf_note_orange.png'
NOTE4 = 'gf_note_yellow.png'

def PCA(M, k=2):
    x1, x2 = M.shape
    if (x1 >= x2): M = M.T
    R = np.cov(M - mean(M))
    lamb, Vy = np.linalg.eigh(R)
    return lamb, M.T.dot(Vy)[:,:k]

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

def set_plt_params(figsize=[8,5], figdpi=150, style='default'):
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
    img = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE)
    notes = [NOTE1, NOTE2, NOTE3, NOTE4]
    A = np.zeros([len(notes), w*h])

    for index, note in enumerate(notes):
        imgc = cv2.imread(note, cv2.IMREAD_GRAYSCALE)
        A[index:w*h] = imgc.reshape(w*h)

    lamb, V = PCA(A.T, k=2)
    U = mean(V.T)

    W, H = (840, 560)
    img_pad = np.zeros((W, H))
    img_pad[:img.shape[0],:img.shape[1]] = img
    j = 0; leng = int(W*H/(w*h))
    count = 0

    print(img_pad.shape)

    plt.ion()

    fig, axs = plt.subplots()
    im1 = axs.imshow(img)

    stepx, stepy = (10, 30)
    maxx = int(W/stepx) - 20
    maxy = int(H/stepy) - 1
    d = np.zeros(maxx*maxy); k = 0

    for i in range(maxx):
        for j in range(maxy):
            dx = (stepx*i, w + (stepx*i))
            dy = (stepy*j, h + (stepy*j))
            im = img_pad[dx[0]:dx[1],dy[0]:dy[1]]

            L = mean(U * (im.reshape(w*h) - mean(A)))
            dist = euclidian_dist(U, L)

            print('D = {:.2f}'.format(dist))
            print('Maping = ({}, {})'.format(i, j))

            d[k] = dist; k += 1
            norm = 1e6
            threshold = 0.93

            if dist < threshold*norm:
                count += 1
                pt = (dy[0], dx[0])

                print('FOUND!')
                print('Coords (x, y) = ({}, {})'.format(pt[0],pt[1]))
                print('------')

                cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (255,255,255), 3)
                im1.set_data(img)
            fig.canvas.flush_events()
            # time.sleep(0.5)

    print('Found {} notes'.format(count))

    # pt = (240,480)
    # cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (255,255,255), 3)
    # cv2.imshow('', img)
    # cv2.waitKey()
