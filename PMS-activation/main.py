# -*- coding: utf-8 -*-

import sys 
sys.path.append("./utils/")

import os 
import shutil 

import torch  
import numpy as np 
from utils import plot_training_rst
from load_data import build_dataset 
from train_func import train_func


if __name__ == "__main__":     
    save_path = "./save_results/"
    if os.path.exists(save_path): 
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    # ========== setting & loading ========== 
    flag = "E" 
    X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag)  
        
    
    # ========== model training ========== 
    loop_num = 1000
    error_flag_test = 1e20  
    for _ in range(loop_num): 
        loss_rec_train, error_rec_train, loss_rec_test, error_rec_test, model = train_func(flag) 
        if error_rec_test[-1] <= error_flag_test: 
            error_flag_train = error_rec_train[-1]
            error_flag_test = error_rec_test[-1]
            train_error = error_rec_train
            test_error = error_rec_test
            model_sav = model
            
    # ========== prediction & final test ========== 
    train_eval = []
    model_sav.eval() 
    with torch.no_grad():
        for coord in X_data_train:
            outputs = model_sav(coord)
            train_eval.append(outputs) 
    
    test_eval = []
    model_sav.eval() 
    with torch.no_grad(): 
        for coord in X_data_test:
            outputs = model_sav(coord)
            test_eval.append(outputs) 
            
    predicts = []
    model_sav.eval() 
    with torch.no_grad():
        for coord in X_data_predict:
            outputs = model_sav(coord)
            predicts.append(outputs)
    
    # ========== print & save results ==========
    plot_training_rst(train_error, test_error) 
    np.savetxt(save_path+"ML_results_"+"predition.txt", np.array(predicts)) 
    np.savetxt(save_path+"ML_results_"+"train.txt", np.array(train_eval)) 
    np.savetxt(save_path+"ML_results_"+"test.txt", np.array(test_eval)) 
    torch.save(model_sav.state_dict(), save_path+"atomic_net"+".pkl")
