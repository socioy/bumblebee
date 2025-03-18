from bumblebee import Mouse
from bumblebee.ai.predict import Predictor
m = Mouse()
p = Predictor()
path = p._predict([100,100], [100,100])

print(path)