# -*- coding: utf-8 -*-

import numpy as np 
import xlrd 
import torch 

def load_data(flag):
    file="./data1.xlsx"
    wb = xlrd.open_workbook(file) 
    sheet = wb.sheet_by_index(0) 
    
    num = 61  
    electronegativity = np.array([sheet.cell_value(loopi+1,1) for loopi in range(num)])
    d_electrons = np.array([sheet.cell_value(loopi+1,2) for loopi in range(num)])          
    group = np.array([sheet.cell_value(loopi+1,3) for loopi in range(num)])              
    radius = np.array([sheet.cell_value(loopi+1,4) for loopi in range(num)])              
    N_numbers = np.array([sheet.cell_value(loopi+1,5) for loopi in range(num)])      
    Ebar = np.array([sheet.cell_value(loopi+1,6) for loopi in range(27)])
    
    
    X_data = np.stack((electronegativity, d_electrons, group, radius, N_numbers), axis=0).T 
    X_data = torch.from_numpy(X_data).float()  

    if flag == "E": 
        y_data = torch.from_numpy(Ebar).float() 

    return X_data, y_data, [electronegativity, d_electrons, group, radius, N_numbers, y_data] 
    
def build_dataset(flag):
    X_data, y_data, _ = load_data(flag)
    
    train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] 
    
    test_list = [2, 5, 14, 20, 25]  

        
    for idx in test_list: 
        train_list.remove(idx)
    
    predict_list = [27+loopi for loopi in range(34)]
    
    X_data_train, y_data_train = [], []
    X_data_test, y_data_test = [], [] 
    X_data_predict = [] 
    for index in range(61):  
        if index in train_list:
            X_data_train.append(X_data[index,:].reshape(1,1,5))
            y_data_train.append(y_data[index].reshape(1)) 
            
        if index in test_list: 
            X_data_test.append(X_data[index,:].reshape(1,1,5))
            y_data_test.append(y_data[index].reshape(1)) 
            
        if index in predict_list:
            X_data_predict.append(X_data[index,:].reshape(1,1,5)) 
            
    return X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict


if __name__ == "__main__": 
    flag = "E" 
    X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag) 
    
    
    
    