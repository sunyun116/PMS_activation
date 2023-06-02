# -*- coding: utf-8 -*-

import torch  
import numpy as np 
from utils import MyError, MyLoss, plot_training_rst
from model import Net 
from load_data import build_dataset, load_data  


# ========== data load ========== 
flag = "E" 
X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag) 
model = Net(flag) 
model.load_state_dict(torch.load("../sav/atomic_net_"+flag+".pkl")) 

predicts = [] 
model.eval() 
with torch.no_grad():
    for coord in X_data_predict:
        outputs = model(coord)
        predicts.append(outputs)

print(predicts) 


from results_analysis import descriptor 

_, _, [electronegativity, d_electrons, group, radius, N_numbers, data1] = load_data(flag) 

desc_ML = []
for E, d, g, r, N in zip(electronegativity, d_electrons, group, radius, N_numbers):
    desc_ML.append(descriptor(E, d, g, r, N))

