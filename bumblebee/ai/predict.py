import numpy as np
import torch
import os
from bumblebee.ai import RNN, Attention


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
        self.MAX_COORDINATE = 4096  # will allow for 4K resolution, taken from training notebook: `rnn-train.ipynb`
        self.MIN_COORDINATE = 0

        # Constant for RNN model
        self.INPUT_DIM = 4  # Start (x, y) and Destination (x, y) coordinates
        self.HIDDEN_DIM = 512
        self.STEPS = 39  # Number of intemediate steps between start and destination
        self.OUTPUT_DIM = 78  # 2 coordinates (x, y) for each of the 39 steps
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

    def normalize(self, data: int | float) -> float:
        """
        Normalize coordinates to [0,1] range for model input
        """
        return (data - self.MIN_COORDINATE) / (
            self.MAX_COORDINATE - self.MIN_COORDINATE
        )

    def denormalize(self, data: int | float) -> float:
        """
        Denormalize coordinates from [0,1] range to original range
        """
        return data * (self.MAX_COORDINATE - self.MIN_COORDINATE) + self.MIN_COORDINATE
