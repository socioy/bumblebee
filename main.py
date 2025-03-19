from bumblebee import Mouse
from bumblebee.ai.predict import Predictor

m = Mouse()
p = Predictor()
path = p.predict([100, 700], [700, 60])
