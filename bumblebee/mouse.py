import random
import time

import numpy as np
import pyautogui

from bumblebee.ai import Predictor


class Mouse:
    def __init__(self):
        self.__predictor = Predictor()
        self.__setup_pyautogui()
        self.SPEED = 1000  # Speed in pixels per second

    def __setup_pyautogui(self):
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0.001
        pyautogui.FAILSAFE = False

    def move(self, destX, destY):
        currentX, currentY = pyautogui.position()

        start = np.array([currentX, currentY], dtype=np.float32)
        destination = np.array([destX, destY], dtype=np.float32)

        path_points = self.__predictor.predict(start, destination)
        x, y, speed_factor = path_points[:, 0], path_points[:, 1], path_points[:, 2]

        del path_points

        total_points = len(x)
        adjacent_distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        adjacent_distances = np.insert(adjacent_distances, 0, 0)
        distance_to_last_position = 0
        for i in range(1, total_points):
            distance = adjacent_distances[i]

            if distance < 20.0:
                distance_to_last_position += distance
                continue

            X = int(x[i])
            Y = int(y[i])

            if distance >= 40.0:
                speed = self.SPEED * random.uniform(1, 1.1) * 1.2
            else:
                speed = self.SPEED * random.uniform(0.95, 1.05)

            distance += distance_to_last_position

            time_to_move = round(((distance / speed) * speed_factor[i]), 4)
            move_duration = 0.0 if time_to_move < 0.1 else time_to_move
            to_sleep_duration = round(
                (time_to_move - move_duration) * random.uniform(1, 1.5), 3
            )
            print(distance, speed, time_to_move)
            start_time = time.time()
            pyautogui.moveTo(X, Y, duration=move_duration)
            pyautogui.sleep(to_sleep_duration)
            print(f"to sleep: {to_sleep_duration}")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            distance_to_last_position = 0
            #
