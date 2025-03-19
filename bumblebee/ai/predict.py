import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from bumblebee.ai.rnn import CursorRNN as RNN


class Predictor:
    def __init__(self, model_name: str = "bumblebee-c-v1.pth"):
        """
        Initialize the prediction model with a pre-trained RNN for generating intermediate path steps.

        Parameters:
            model_name (str): The filename of the pre-trained model to load. Defaults to "bumblebee-c-v1.pth".

        Attributes:
            MAX_COORDINATE (int): Maximum coordinate value supported (e.g., 4096 for 4K resolution).
            MIN_COORDINATE (int): Minimum coordinate value.
            INPUT_DIM (int): Number of input features (start and destination x, y coordinates).
            HIDDEN_DIM (int): Size of the hidden state in the RNN.
            STEPS (int): Number of intermediate steps predicted between the start and destination.
            OUTPUT_DIM (int): Number of output features (2 coordinates for each intermediate step).
            LSTM_LAYERS (int): The number of LSTM layers in the RNN.
            device (torch.device): The device (CUDA, MPS, or CPU) on which the model computations are performed.
            model (RNN): The RNN model instance used for predicting paths.
            model_path (str): The full file path to the pre-trained model file.

        The constructor loads the pre-trained model, maps it to the appropriate computation device,
        and sets the model to evaluation mode.
        """
        # Constants for normalization
        self.MAX_COORDINATE = 1920  # max resolution used in data collection was 1920x1080p so 1980 is used in training process,taken from training notebook: `rnn-train.ipynb`
        self.MIN_COORDINATE = 0

        # Constant for RNN model
        self.INPUT_DIM = 4  # Start (x, y) and Destination (x, y) coordinates
        self.HIDDEN_DIM = 512
        self.STEPS = 39  # Number of intemediate steps between start and destination
        self.OUTPUT_DIM = self.STEPS * 2  # 2 coordinates (x, y) for each steps
        self.LSTM_LAYERS = 2

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )  # Check for best hardware to run the model on

        self.model = RNN(
            self.INPUT_DIM,
            self.HIDDEN_DIM,
            self.OUTPUT_DIM,
            num_layers=self.LSTM_LAYERS,
        ).to(self.device)

        self.model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
        if self.device == "cuda":
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        self.model.eval()

    def __normalize(self, data: int | float) -> float:
        """
        Normalize coordinates to [0,1] range for model input
        """
        return (data - self.MIN_COORDINATE) / (
            self.MAX_COORDINATE - self.MIN_COORDINATE
        )

    def __denormalize(self, data: int | float) -> float:
        """
        Denormalize coordinates from [0,1] range to original range
        """
        return data * (self.MAX_COORDINATE - self.MIN_COORDINATE) + self.MIN_COORDINATE

    def __calculate_distance(self, point1: list, point2: list) -> float:
        """
        Calculate the Euclidean distance between two points in 2D space.
        Parameters:
            point1 (list): The first point as [x1, y1].
            point2 (list): The second point as [x2, y2].
        Returns:
            float: The Euclidean distance between point1 and point2.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def __clean_path(self, path: np.ndarray, min_distance=10) -> np.ndarray:
        """
        Clean the predicted path by removing points that are too close to each other.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
            min_distance (float): The minimum distance between consecutive points to keep them.
        Returns:
            np.ndarray: The cleaned path with points that are sufficiently spaced apart.
        """
        cleaned_path = [path[0]]  # Start with the first point
        for i in range(1, len(path)):
            if self.__calculate_distance(cleaned_path[-1], path[i]) >= min_distance:
                cleaned_path.append(path[i])

        return np.array(cleaned_path)

    def __calculate_speed_factor(self, progress: float) -> float:
        """
        Calculate the speed factor based on the progress of the path.
        Parameters:
            progress (float): The progress of the path as a value between 0 and 1.
        Returns:
            float: The speed factor, which is higher for later progress in the path.
        """
        return 1.5 - np.cos(
            progress * np.pi
        )  # slower at start and end, faster in the middle

    def __smooth_path(self, path: np.ndarray, noise_factor=0.03) -> np.ndarray:
        """
        Smooth the predicted path by applying Gaussian noise to the coordinates.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
            noise_factor (float): The standard deviation of the Gaussian noise to be added.
        Returns:
            np.ndarray: The smoothed path with Gaussian noise applied to the coordinates.
        """
        smoothed_x = gaussian_filter1d(
            path[:, 0], sigma=1
        )  # apply gaussian filter to x coordinates
        smoothed_y = gaussian_filter1d(
            path[:, 1], sigma=1
        )  # apply gaussian filter to y coordinates

        noise_x = np.random.normal(
            0, noise_factor, size=smoothed_x.shape
        )  # generate noise for x coordinates
        noise_y = np.random.normal(
            0, noise_factor, size=smoothed_y.shape
        )  # generate noise for y coordinates

        t = np.linspace(0, 1, len(smoothed_x))  # Normalized time variable
        noise_decay = np.exp(-4 * t)  # Exponential decay function

        # Here we are adding noise decay so that noise reduces as we move from start to end

        smoothed_x += noise_x * noise_decay
        smoothed_y += noise_y * noise_decay

        # Apply final smoothing to prevent abrupt jumps
        smoothed_x = gaussian_filter1d(smoothed_x, sigma=0.5)
        smoothed_y = gaussian_filter1d(smoothed_y, sigma=0.5)
        return np.column_stack((smoothed_x, smoothed_y))

    def _predict(
        self, start: list[int, int], destination: list[int, int]
    ) -> np.ndarray:
        """
        Predict the intermediate path steps between the start and destination coordinates.

        Parameters:
            start (list[int, int]): The starting coordinate as [x, y].
            destination (list[int, int]): The destination coordinate as [x, y].

        Returns:
            np.ndarray: The predicted intermediate steps of the path.

        This method uses the pre-trained RNN model to generate intermediate steps between the start
        and destination. The input coordinates are normalized before being fed into the model.
        """
        # Combine start and destination coordinates into a single list
        input_data = start + destination

        # Convert the combined coordinates list to a NumPy array with float32 type for precision
        input_arr = np.array(
            [input_data], dtype=np.float32
        )  # Don't try to remove [] from here, it will break the code

        # Normalize the input array to be within the [0, 1] range for model compatibility
        input_normalized_arr = self.__normalize(input_arr)
        print(input_normalized_arr)

        # Convert the normalized array to a PyTorch tensor, send it to the appropriate device, and add a dimension for batch processing (unsqueeze simulates a batch of size 1)
        input_tensor = torch.tensor(input_normalized_arr).to(self.device).unsqueeze(1)

        # Clean up intermediate arrays to free memory
        del input_arr
        del input_normalized_arr

        with torch.no_grad():
            # Pass the input tensor through the model to get the output
            output = self.model(input_tensor)

        # Clean up the input tensor to free memory
        del input_tensor

        # Reshape the output to match the expected dimensions and move it to CPU
        output = output[0].view([self.STEPS, 2]).cpu().numpy()

        # Denormalize the output to convert it back to the original coordinate range
        output = self.__denormalize(output)

        return output
