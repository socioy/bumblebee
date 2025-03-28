
    def type(self, text, isHuman=False):
        """
        Type the given text with human-like rhythm, pauses, and occasional typos.
        Simulates realistic typing behavior based on character complexity and keyboard layout.
        """
        if not isHuman:
            pyautogui.typewrite(text)
            return

        # todo warn user if caps lock is on through telegram
        for i, char in enumerate(text):
            # Calculate base delay from typing speed (CPM to seconds)
            base_delay = 60 / self.typing_speed

            # Adjust timing based on character complexity
            char_complexity = 1.0
            if char in string.punctuation:
                char_complexity = 1.3  # Punctuation takes longer to type
            elif char.isupper():
                char_complexity = 1.2  # Uppercase takes longer to type

            # Factor in distance from previous key
            distance_factor = 0
            if self.last_char:
                distance = self._key_distance(self.last_char, char)
                distance_factor = distance * 0.05  # Longer distances take more time

            # Add realistic variation to typing speed
            consistency_variation = random.uniform(
                1 - self.consistency, 1 + self.consistency
            )

            # Add pauses for natural reading/thinking rhythm
            context_pause = 0
            shift_delay = 0.1 if self._is_shift_needed(char) else 0

            # Add pauses after sentence endings and commas
            if char in ".?!":
                context_pause = random.uniform(0.2, 0.4)  # Pause at end of sentence
            elif char in ",":
                context_pause = random.uniform(0.05, 0.2)  # Shorter pause at comma
            elif char == " " and i > 0 and text[i - 1] in ".!?":
                context_pause = random.uniform(0.2, 0.5)  # Pause after end of sentence

            # Calculate final delay for this character
            delay = (
                (
                    base_delay
                    * char_complexity
                    * (1 + distance_factor)
                    * consistency_variation
                    * self.typing_rhythm
                )
                / 5
            ) + context_pause

            # Simulate occasional typos for realism
            if random.random() < 0.03:  # 3% chance of typo
                typo_char = random.choice(string.ascii_lowercase)
                pyautogui.typewrite(typo_char)
                time.sleep(random.uniform(0.1, 0.2))  # Brief pause before correction
                pyautogui.typewrite("\b")  # Backspace to correct the typo
                time.sleep(
                    random.uniform(0.05, 0.1)
                )  # Pause before typing correct char

            # Handle uppercase characters with shift key
            if char.isupper():
                pyautogui.keyDown("shift")
                pyautogui.typewrite(char.lower())
                time.sleep(shift_delay)
                pyautogui.keyUp("shift")
            else:
                pyautogui.typewrite(char)

            self.last_char = char  # Remember last character for distance calculation

            # Wait before typing the next character
            time.sleep(delay)

    def shortcut(self, *keys):
        for key in keys[:-1]:
            pyautogui.keyDown(key)
        pyautogui.press(keys[-1])
        for key in reversed(keys[:-1]):
            pyautogui.keyUp(key)

    def press(self, key):
        pyautogui.press(key)

    def copy(self):
        if sys.platform == "darwin":
            self.shortcut("command", "c")
        else:
            self.shortcut("ctrl", "c")

    def paste(self):
        if sys.platform == "darwin":
            self.shortcut("command", "v")
        else:
            self.shortcut("ctrl", "v")
