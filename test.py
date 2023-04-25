import mlflow
import pandas as pd

from common_functions import load_data, data_cleaning, preprocessing

logged_model = 'runs:/884426e2162c43e0aa353ed8e089575b/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


def prepare_dat_for_prediction(path):
    data = load_data(path)
    cleaned_data = data_cleaning(data)
    final_data = preprocessing(cleaned_data)
    X = final_data.loc[:, final_data.columns != 'y']
    return X


# Predict on a Pandas DataFrame.
data = prepare_dat_for_prediction('banking.csv')

res = loaded_model.predict(pd.DataFrame(data))
print(0)
