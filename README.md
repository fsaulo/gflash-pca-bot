## Computer Vision application using PCA

Principal Component Analysis applied on low frame rate video to detect objects as it moves through a platform.

## Instructions

The screenshot module does not work with systems other than those taht uses X11 window manager. Support will (probably) be added in the future.
You can currently change manually to use pyscreenshot module that works in any system in exchange that is a lot slower.

Replace the following line from 'guitarbot.py'

```python
img = screenshot.grab_screen(x0, y0, res[0] + x0, res[1] + y0)
```

With:

```python
img = pyscreenshot.grab(bbox=(x0, y0, res[0] + x0, res[1] + y0))
```

Making appropriate imports

```python
import pyscreenshot
```
