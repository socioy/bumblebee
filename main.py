from bumblebee import Mouse
from bumblebee.ai.predict import Predictor

m = Mouse()
p = Predictor()
path = p._predict([59, 800], [90, 129])

print(path)
