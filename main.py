from bumblebee import Mouse
import pyautogui
import time

x, y = pyautogui.size()
import random
m = Mouse()
pyautogui.move(random.randint(0, x), random.randint(0, y))
for i in range(100):
    m.move(random.randint(0, x), random.randint(0, y))
    time.sleep(2)
# p = Predictor()
# path = p.predict([100, 700], [700, 60])
