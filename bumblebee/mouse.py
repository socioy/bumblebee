import pyautogui
import torch
import torch.nn as nn

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
        path_points = self.__predictor.predict([currentX, currentY], [destX, destY])
        for point in path_points:
            pyautogui.moveTo(point[0], point[1], duration=0.1)
