import numpy as np
import pandas as pd


def load_data(path):
    data = pd.read_csv(path)
    return data


def data_cleaning(data):
    print("na values available in data \n")
    print(data.isna().sum())
    data = data.dropna()
    print("after droping na values \n")
    print(data.isna().sum())
    return data


def preprocessing(data):
    data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])

    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'poutcome']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1 = data.join(cat_list)
        data = data1

    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'poutcome']
    data_vars = data.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]

    final_data = data[to_keep]

    final_data.columns = final_data.columns.str.replace('.', '_')
    final_data.columns = final_data.columns.str.replace(' ', '_')
    return final_data


def create_experiment(experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                      roc_auc_plot_path=None, run_params=None):
    import mlflow
    # mlflow.set_tracking_uri("http://localhost:5000")
    # use above line if you want to use any database like sqlite as backend storage for model else comment this line
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        if not run_params is None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        if not confusion_matrix_path is None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')

        if not roc_auc_plot_path is None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")

        mlflow.set_tag("tag1", "Iris Classifier")
        mlflow.set_tags({"tag2": "Logistic Regression", "tag3": "Multi classification using Ovr - One vs rest class"})
        mlflow.sklearn.log_model(model, "model")
    print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))
