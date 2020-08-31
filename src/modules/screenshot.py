import ctypes
import os
from PIL import Image
import time

LibName = 'prtscn.so'
AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
grab = ctypes.CDLL(AbsLibPath)

def grab_screen(x1,y1,x2,y2):
    w, h = x2-x1, y2-y1
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1,y1, w, h, result)
    return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)

def save(img, path=''):
    path = '/home/felix/Documents/Projects/gflash/images/screenshots/'
    img.save(path + 'screenshot_{}.png'.format(int(time.time())))
    print('Screenshot taken saved at ' + path)


if __name__ == '__main__':
    start = time.time()
    im = grab_screen(0,0,1440,900)
    print(im)
    print(time.time() - start)
    im.show()
