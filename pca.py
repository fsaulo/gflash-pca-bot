import numpy as np
import matplotlib.pyplot as plt
import cv2

STAGE = 'gf_stage.png'
NOTE1 = 'gf_note_blue.png'
NOTE2 = 'gf_note_red.png'
NOTE3 = 'gf_note_orange.png'
NOTE4 = 'gf_note_yellow.png'

def PCA(M, k=2):
    x1, x2 = M.shape
    if (x1 >= x2): M = M.T
    R = np.cov(M)
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

def set_plt_params(figsize=[10,4], figdpi=150, style='default'):
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

    set_plt_params(style='seaborn-darkgrid')

    w, h = (60, 80)
    img = cv2.imread(STAGE, cv2.IMREAD_GRAYSCALE)
    notes = [NOTE1, NOTE2, NOTE3, NOTE4]
    A = np.zeros([len(notes), w*h])

    for index, note in enumerate(notes):
        imgc = cv2.imread(note, cv2.IMREAD_GRAYSCALE)
        A[index:w*h] = imgc.reshape(w*h)

    lamb, V = PCA(A.T, k=4)

    U = normalize(mean(V.T * A))

    W, H = (840, 560)
    img_pad = np.zeros((W, H))
    img_pad[:img.shape[0],:img.shape[1]] = img
    j = 0; leng = int(W*H/(w*h))
    count = 0

    print(img_pad.shape)

    for j in range(0, 7):
        for i in range(0, 14):
            im = img_pad[w*i:w*(i+1),j*h:h*(j+1)]
            L = normalize(mean(V.T * im.reshape(w*h)))
            dist = euclidian_dist(U, L)

            print('D = {:.2f}'.format(dist))
            print('index = ({}, {})'.format(i, j))

            if dist < 0.35:
                count += 1
                print('FOUND!')
                print('------')


    print('Found {} notes'.format(count))

    # imshow(adjust_to_plot(V, 4, (w, h)))

    # img_rec1 = np.dot(V1, V1.T) * np.matrix(template)
    # img_rec2 = np.dot(V2, V2.T) * np.matrix(area)

    # print('---------------PCA---------------')
    # print('Mean = {:.2f}'.format(R.mean()))
    # print('Diag = {:.2f}'.format(diagonal_mean))
    # print('Weight = {:.2f}'.format(weight))
    # print('Max = {:.2f}'.format(np.max(R)))

    plt.show()
