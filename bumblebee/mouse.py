import random
import time
from typing import Any

import numpy as np
import pyautogui

from bumblebee.ai import Predictor


class Mouse:
    def __init__(self):
        self.__predictor = (
            Predictor()
        )  # predictor instance for AI prediction of mouse path
        self.__setup_pyautogui()  # setup pyautogui for mouse movement
        """
        Base mouse movement speed in pixels/second. Has greater impact over longer distances.
        For shorter movements, speed differences are less noticeable but still maintained using sleeping techniques.
        """
        self.__SPEED = 2000 # 2000 pixels/second feels natural

    def __setup_pyautogui(self):
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0.001  # setting the pause time of pyautogui in each move to 1ms; default is 100ms
        pyautogui.FAILSAFE = False  # required to allow mouse to move to corners

    def __prepare_data_for_move(self, path_points: np.ndarray) -> np.ndarray:
        """
        Processes path points data to generate mouse movement commands.

        Takes a NumPy array containing path coordinates with x, y positions and speed scaling
        factors for each point. Analyzes distances between points and calculates appropriate
        movement timing parameters for natural-looking cursor motion.

        Args:
            path_points (np.ndarray): 2D array of shape (N,3) containing:
                - Column 0: X coordinates
                - Column 1: Y coordinates
                - Column 2: Speed modifier factors

        Returns:
            np.ndarray: 2D array of shape (M,4) containing:
                - Column 0: Target X position (pixels)
                - Column 1: Target Y position (pixels)
                - Column 2: Duration of movement (seconds)
                - Column 3: Sleep delay after movement (seconds)

        Note: M may be less than N since some points are filtered out if too close together.

        """

        x, y, speed_factor = path_points[:, 0], path_points[:, 1], path_points[:, 2]
        total_points = len(x)

        adjacent_distances = np.sqrt(
            np.diff(x) ** 2 + np.diff(y) ** 2
        )  # Calculate the distance between adjacent points, arr[next_index] - array[current_index
        adjacent_distances = np.insert(
            adjacent_distances, 0, 0
        )  # Insert a zero at the beginning of the array, to handle the first point; first point has no previous point
        distance_to_last_position = 0  # keep track of the distance to the last position if the distance is less than 5px
        data_required_for_move = np.array([])

        for i in range(1, total_points):
            distance = adjacent_distances[i]

            """
            Skip the point, if its distance to the previous point is less than 5px.
            For most small distances, pyautogui will take 0seconds to move the mouse.
            So, we can skip the point to avoid small movements making the mouse jittery.
            """
            if (
                distance < 5.0
            ):  # skip the point if distance between it and the previous point is less than 5px; doing it to avoid small movements making the mouse jittery
                distance_to_last_position += distance
                continue

            X = int(x[i])
            Y = int(y[i])

            if distance >= 40.0:  # increase the speed if distance is greater than 40px
                speed = self.__SPEED * random.uniform(1, 1.1) * 1.2
            else:
                speed = self.__SPEED * random.uniform(
                    0.95, 1.05
                )  # randomize the speed to make it more natural

            distance += distance_to_last_position

            time_to_move = round(((distance / speed) * speed_factor[i]), 4)
            move_duration = (
                0.0 if time_to_move < 0.1 else time_to_move
            )  # the time <= 0.1 is considered 0.0 by pyautogui, so we will use sleep method to sleep for duration < 0.1s manually
            to_sleep_duration = round((time_to_move - move_duration), 6)
            distance_to_last_position = 0
            data_required_for_move = np.append(
                data_required_for_move, [X, Y, move_duration, to_sleep_duration]
            )

        return data_required_for_move.reshape(-1, 4)

    def __assert_valid_button(self, button:str):
        assert isinstance(button, str), "Invalid button type:{}".format(type(button))
        assert button in [
            "left",
            "middle",
            "right",
            "primary",
            "secondary",
        ], "Invalid button:{}. Button must be one of 'left', 'middle', 'right', 'primary', 'secondary'.".format(button)

    def hold(self, button: Any):
        """
        Simulates holding down a mouse button.

        Args:
            button (str): The mouse button to hold. Options: 'left', 'middle', 'right', 'primary', 'secondary'
        """
        self.__assert_valid_button(button)
        time.sleep(random.uniform(0.05, 0.1))
        pyautogui.mouseDown(button=button)

    def release(self, button: Any):
        """
        Simulates releasing a mouse button.

        Args:
            button (str): The mouse button to release. Options: 'left', 'middle', 'right', 'primary', 'secondary'
        """
        self.__assert_valid_button(button)
        time.sleep(random.uniform(0.05, 0.1))
        pyautogui.mouseUp(button=button)

    def drag_to(self, destX, destY, button:str = "left"):
        """
        Simulates dragging the mouse to a destination. Utilizes native PyAutoGUI functionality, but with a custom speed prediction.
        Unlike move() method, this method does not simulate human behavior.

        Args:
            destX (int): The x-coordinate of the destination.
            destY (int): The y-coordinate of the destination.
            button (str): The mouse button to drag with. Defaults to 'left'. Options: 'left', 'middle', 'right', 'primary', 'secondary'.
        Raises:
            AssertionError: If the button is not valid.
        """
        self.__assert_valid_button(button)
        current_coordinates = np.array(pyautogui.position())
        dest_coordinates = np.array([destX, destY])
        distance = self.__predictor._calculate_distance(current_coordinates, dest_coordinates)
        pyautogui.dragTo(destX, destY, duration=distance / self.__SPEED, button=button)
        time.sleep(random.uniform(0.05, 0.1))

    def hover(self, radius:int, duration:float):
        currentX, currentY = pyautogui.position()
        start_time = time.time()
        while True:
            if time.time() - start_time > duration:
                break
            #todo: implement hover
            self.move(currentX + random.uniform(-radius, radius), currentY + random.uniform(-radius, radius))

    def click(self, button: Any = "left"):
        """
        Simulates a mouse click at the current position.

        Args:
            button (str): The mouse button to click. Defaults to 'left'. Options: 'left', 'middle', 'right', 'primary', 'secondary'
        """
        self.__assert_valid_button(button)
        time.sleep(random.uniform(0.05, 0.1))
        pyautogui.click(button=button)

    def move(self, destX, destY, drag=False):
        currentX, currentY = pyautogui.position()

        start = np.array([currentX, currentY], dtype=np.float32)
        destination = np.array([destX, destY], dtype=np.float32)

        path_points = self.__predictor.predict(start, destination)

        """
        Calculate movement parameters for each point (speed, distance, duration, sleep time)
        before executing movement to prevent lag during mouse movement.
        We prepare all calculations first via __prepare_data_for_move() rather than
        calculating during movement to ensure smooth cursor motion.
        """
        path_data = self.__prepare_data_for_move(path_points)

        del path_points

        for data in path_data:
            X, Y, move_duration, sleep_duration = data
            if drag:
                pyautogui.dragTo(X, Y, duration=move_duration, button="left")
            else:
                pyautogui.moveTo(X, Y, duration=move_duration)
            pyautogui.sleep(sleep_duration)

    def move_to_and_click(self, destX, destY, button: Any = "left"):
        """
        Move the mouse cursor to the specified coordinates and click.
        Args:
            destX (int): The x-coordinate of the destination.
            destY (int): The y-coordinate of the destination.
            button (str): The button to click. Defaults to "left".

        Raises:
            AssertionError: If the button is not "left", "middle", "right", "primary", or "secondary".
        """
        self.move(destX, destY)
        time.sleep(random.uniform(0.5, 2))
        self.click(button)

    def set_speed(self, speed: float):
        """
        Set the speed of the mouse cursor.
        Args:
            speed (float): The speed of the mouse cursor. The speed is measured in pixels per second.

        Raises:
            AssertionError: If the speed is not greater than 0.
            AssertionError: If the speed is not a float.
        """
        assert speed > 0, "Speed must be greater than 0"
        assert isinstance(speed, float), "Speed must be a float"
        self.__SPEED = speed
