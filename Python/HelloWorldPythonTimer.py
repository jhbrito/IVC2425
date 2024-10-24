from threading import Timer
import time

running = False

def func():
    global running
    print("running")
    running = False


while True:
    if not running:
        running = True
        print("starting")
        t = Timer(interval=5.0, function=func)
        t.start()
    else:
        print("already started")
    time.sleep(1)