import pymouse as ps

m = ps.PyMouse()

while True:
    print(m.position(), end='\r')
