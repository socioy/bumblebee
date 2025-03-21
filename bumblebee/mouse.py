import numpy as np
import pyautogui

from bumblebee.ai import Predictor


class Mouse:
    def __init__(self):
        self.__predictor = Predictor()
        self.__setup_pyautogui()
        pass

    def __setup_pyautogui(self):
        pyautogui.FAILSAFE = False

    def move(self, destX, destY):
        currentX, currentY = pyautogui.position()
        start = np.array([currentX, currentY], dtype=np.float32)
        destination = np.array([destX, destY], dtype=np.float32)
        path_points = self.__predictor.predict(start, destination)
        for point in path_points:
            pyautogui.moveTo(point[0], point[1], duration=0.1)
