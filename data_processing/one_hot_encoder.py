import os
import pandas as pd
import torch
import numpy
from tqdm import tqdm
import pickle

params = [
    'user_edited',
    'parameters_tone',
    'parameters_switchboard_template_name',
    'parameters_theme',
    'parameters_prompt_template_name',
    'has_logo',
]

SAVE_PATH = "one_hot_encoding.pkl"

def one_hot_encode_legacy(dataframe : pd.DataFrame) -> torch.Tensor:

    allsets = [[] for _ in params]

    all_one_hots = []

    print('gathering unique points')

    for index, row in dataframe.iterrows():
        for i in range(len(params)):
            if (row[params[i]] not in allsets[i]):
                allsets[i].append(row[params[i]])
    
    print('one_hot_encoding variables')

    training_bar = tqdm(total=dataframe.shape[0], desc="Extracting training text", unit="image")
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
        training_bar.update(1)
        
    combined_one_hot = all_one_hots

    return combined_one_hot

def one_hot_encode(dataframe: pd.DataFrame) -> list:
    if os.path.exists(SAVE_PATH):
        print('OHE: loading saved one-hot-encoding')
        with open(SAVE_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        param_names = saved_data["param_names"]
        category_names = saved_data["category_names"]
        print(param_names)
        print(category_names)
    else:
        print('OHE: could not find save file, generating OHE')
        param_names = sorted(params)
        print(param_names)
        category_names = {
            param: sorted(filter(
                lambda x : isinstance(x, str) or isinstance(x, numpy.bool_), 
                dataframe[param].unique()
            )) 
                for param in param_names
        }
        print(category_names)
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump({"param_names": param_names, "category_names": category_names}, f)

    encoded_tensors = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Encoding Progress"):
        encoding = []
        for param in param_names:
            param_value = row[param]
            unique_categories = category_names[param]
            one_hot = [0] * len(unique_categories)
            for i, category in enumerate(unique_categories):
                if category == param_value:
                    one_hot[i] = 1
                    break
            encoding.extend(one_hot)
                
        encoded_tensors.append(torch.tensor(encoding, dtype=torch.int))
    
    return encoded_tensors