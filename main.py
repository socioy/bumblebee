import time

import pyautogui

from bumblebee import Mouse

x, y = pyautogui.size()
import random

m = Mouse()
# pyautogui.move(random.randint(0, x), random.randint(0, y))
for i in range(100):
    m.move(800, 300)
    time.sleep(2)
    m.drag_to(800, 500)
    time.sleep(4)
# p = Predictor()
# path = p.predict([100, 700], [700, 60])
