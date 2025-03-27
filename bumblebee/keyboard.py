import platform

import pyautogui
from scipy.sparse import base


class Keyboard:
    def __init__(self, typing_speed=100, consistency=95):
        """
        Initialize the Keyboard class with typing speed and consistency.

        Args:
            typing_speed (int): The speed as percentage of what bumblebee types normally.
            consistency (int): The consistency of typing of bumblebee, should be between 0 and 100.
        """
        assert isinstance(typing_speed, int) or isinstance(
            typing_speed, float
        ), "typing_speed must be an integer or float"
        assert isinstance(consistency, int) or isinstance(
            consistency, float
        ), "consistency must be an integer or float"
        assert 0 <= consistency <= 100, "consistency must be between 0 and 100"

        self.__platform = platform.system()
        self.__typing_speed = typing_speed / 100
        self.__consistency = consistency / 100
        self.__typo = self.__build_typo()
        self.__shift_needed_chars = None
        self.__key_map = self.__build_key_map()

    def __build_typo(self):
        """
        Build the typo map for the keyboard.

        This method initializes the common typo map for the keyboard.
        """
        typo_map = {
            "a": ["s", "q", "w"],
            "s": ["a", "d", "w", "x"],
            "d": ["s", "f", "e", "r"],
            "f": ["d", "g", "r", "t"],
            "g": ["f", "h", "t", "y"],
            "h": ["g", "j", "y", "u"],
            "j": ["h", "k", "u", "i"],
            "k": ["j", "l", "i", "o"],
            "l": ["k", "m", "o", "p"],
            "m": ["l", "n", "p"],
            "n": ["m", "b"],
            "b": ["n", "v"],
            "v": ["b", "c"],
            "c": ["v", "x"],
            "x": ["c", "z"],
            "z": ["x"],
            "1": ["2", "q"],
            "2": ["1", "3", "w"],
            "3": ["2", "4", "e"],
            "4": ["3", "5", "r"],
            "5": ["4", "6", "t"],
            "6": ["5", "7", "y"],
            "7": ["6", "8", "u"],
            "8": ["7", "9", "i"],
            "9": ["8", "0", "o"],
            "0": ["9", "-", "p"],
            "-": ["0", "="],
            "=": ["-"],
            "q": ["w", "a"],
            "w": ["q", "e", "s"],
            "e": ["w", "r", "d"],
            "r": ["e", "t", "f"],
            "t": ["r", "y", "g"],
            "y": ["t", "u", "h"],
            "u": ["y", "i", "j"],
            "i": ["u", "o", "k"],
            "o": ["i", "p", "l"],
            "p": ["o", "["],
        }
        return typo_map

    def __build_key_map(self):
        """
        Build the key map for the keyboard.

        This method creates and returns a dictionary mapping all keys to their positions on a standard QWERTY keyboard layout.
        """

        keyboard_layout = [
            "`1234567890-=",
            " qwertyuiop[]\\",
            " asdfghjkl;'",
            " zxcvbnm,./",
            " ",  # space
        ]  # standard QWERTY keyboard layout

        shift_chars = {
            "a": "A",
            "b": "B",
            "c": "C",
            "d": "D",
            "e": "E",
            "f": "F",
            "g": "G",
            "h": "H",
            "i": "I",
            "j": "J",
            "k": "K",
            "l": "L",
            "m": "M",
            "n": "N",
            "o": "O",
            "p": "P",
            "q": "Q",
            "r": "R",
            "s": "S",
            "t": "T",
            "u": "U",
            "v": "V",
            "w": "W",
            "x": "X",
            "y": "Y",
            "z": "Z",
            " ": " ",
            "1": "!",
            "2": "@",
            "3": "#",
            "4": "$",
            "5": "%",
            "6": "^",
            "7": "&",
            "8": "*",
            "9": "(",
            "0": ")",
            "-": "_",
            "=": "+",
            "[": "{",
            "]": "}",
            "\\": "|",
            ";": ":",
            "'": '"',
            ",": "<",
            ".": ">",
            "/": "?",
        }  # characters that need shift key to be pressed
        self.__shift_needed_chars = [x for _, x in shift_chars.items()]

        key_positions = {}

        for row, keys in enumerate(keyboard_layout):
            for col, key in enumerate(keys):
                key_positions[key] = (row, col)

        for shift_char, base_char in shift_chars.items():
            if base_char in key_positions:
                key_positions[shift_char] = key_positions[base_char]

        return key_positions

    def set_consistency(self, new_consistency):
        """
        Change the consistency of typing.

        Args:
            new_consistency (int): The new consistency of typing, should be between 0 and 100.
        """
        assert isinstance(new_consistency, int) or isinstance(
            new_consistency, float
        ), "new_consistency must be an integer or float"
        assert 0 <= new_consistency <= 100, "new_consistency must be between 0 and 100"

        self.__consistency = new_consistency / 100

    def set_speed(self, new_speed):
        """
        Change the speed of typing.

        Args:
            new_speed (int): The new speed of typing, should be between 0 and 100.
        """
        assert isinstance(new_speed, int) or isinstance(
            new_speed, float
        ), "new_speed must be an integer or float"
        assert 0 <= new_speed <= 100, "new_speed must be between 0 and 100"

        self.__typing_speed = new_speed / 100
