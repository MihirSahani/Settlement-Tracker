import fastai
from fastai.vision.all import *
learn_inf = load_learner('model.pkl')

from fastai.vision.widgets import *


img1 = PILImage.create("/home/krakenmare/Documents/Machine Learning/Resources/Land Classifier Testdata/sentinel-2/2019-07-16-Sentinel-2_L1C_Mesero.jpg")

pred,pred_idx,probs = learn_inf.predict(img1)
lbl_pred = widgets.Label()
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
