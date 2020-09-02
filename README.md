## Visual computing application using PCA

Experimental bot that plays guitar flash (not yet). It uses Principal Component Analysis on screen-shots being taken at very low frame rate to detect the notes as it moves through the stage.
That kind of analysis is not necessary, perhaps extremely redundant for this application, nevertheless it proves a point.


## Instructions

The screenshot module does not work with systems other than uses X11 window manager. Support will be added in the future.
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
