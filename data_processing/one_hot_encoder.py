import os
import pandas as pd
import torch

params = [7, 8, 9, 10, 11, 12]

#train_data_path = "./social-media-post-approval-prediction-with-marky/small_train.csv"
#test_data_path = "./social-media-post-approval-prediction-with-marky/test.csv"

# Dataframes
#train = pd.read_csv(train_data_path)
#test = pd.read_csv(test_data_path)

def one_hot_encode(dataframe) -> torch.Tensor:

    allsets = [[] for _ in params]

    all_one_hots = []

    print('gathering unique points')

    for index, row in dataframe.iterrows():
        for i in range(len(params)):
            if (row[params[i]] not in allsets[i]):
                allsets[i].append(row[params[i]])
    
    print('one_hot_encoding variables')

    for index, row in dataframe.iterrows():
        line_one_hots = []
        for i in range(len(params)):
            one_hot = [0 for _ in allsets[i]]
            where_it_is = 0
            for j in range(len(allsets[i])):
                if (allsets[i][j] == row[params[i]]):
                    where_it_is = j
                    break
            
            one_hot[where_it_is] = 1
            line_one_hots.extend(one_hot)
        
        all_one_hots.append(torch.tensor(line_one_hots))
        if index % 300 == 0:
            print(f'row {index}')
        
    combined_one_hot = all_one_hots

    return combined_one_hot