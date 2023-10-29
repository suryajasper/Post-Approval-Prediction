import pandas as pd
import numpy as np

test_data_path = "./social-media-post-approval-prediction-with-marky/test.csv"

test = pd.read_csv(test_data_path)

# Create a new DataFrame with the first column from the original DataFrame
test_with_answers = test.iloc[:, [0]].copy()

test_with_answers['approved'] = np.random.choice(['true', 'false'], size=test_with_answers.shape[0], p=[0.1, 0.9])

# Display the new DataFrame
#print(test_with_answers)

test_with_answers.to_csv('marky_best_answers_ever.csv', index=False)