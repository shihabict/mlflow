import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

from common_functions import create_experiment


# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000


class MLFlowModelServer:
    def load_data(self, url):
        import pandas as pd
        # Load dataset
        data = pd.read_csv(filepath_or_buffer=url, sep=',')
        return data

    def train_test_split(self, final_data, target_column):
        from sklearn.model_selection import train_test_split
        X = final_data.loc[:, final_data.columns != target_column]
        y = final_data.loc[:, final_data.columns == target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)
        return X_train, X_test, y_train, y_test

    def training_basic_classifier(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        return classifier

    def predict_on_test_data(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def predict_prob_on_test_data(self, model, X_test):
        y_pred = model.predict_proba(X_test)
        return y_pred

    def get_metrics(self, y_true, y_pred, y_pred_prob):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        entropy = log_loss(y_true, y_pred_prob)
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2),
                'entropy': round(entropy, 2)}

    def create_roc_auc_plot(self, clf, X_data, y_data):
        import matplotlib.pyplot as plt
        # from sklearn import metrics
        # metrics.plot_roc_curve(clf, X_data, y_data)
        # X_data = X_data['class'].to_list()
        X_data = X_data['class'].to_numpy()
        RocCurveDisplay.from_estimator(clf, X_data, y_data)
        plt.savefig('roc_auc_curve.png')
        plt.show()

    def create_confusion_matrix_plot(self, clf, X_test, y_test):
        import matplotlib.pyplot as plt
        # from sklearn.metrics import plot_confusion_matrix
        cm = confusion_matrix(X_test, y_test, labels=clf.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()

        # plot_confusion_matrix(clf, X_test, y_test)
        plt.savefig('report/confusion_matrix.png')
        plt.show()

    # Define create_experiment function to track your model experiment within MLFlow

    def train_iris(self):
        url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
        data = self.load_data(url)
        # data.head()
        target_column = 'class'
        X_train, X_test, y_train, y_test = self.train_test_split(data, target_column)

        model = self.training_basic_classifier(X_train, y_train)

        print(f"See the prediction outcome")
        y_pred = self.predict_on_test_data(model, X_test)
        print(y_pred)
        y_pred_prob = self.predict_prob_on_test_data(model, X_test)
        print(y_pred_prob)
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)

        # self.create_roc_auc_plot(model, y_test, y_pred)
        self.create_confusion_matrix_plot(model, y_test, y_pred)

        from datetime import datetime
        experiment_name = "iris_classifier_" + str(datetime.now().strftime("%d-%m-%y"))  ##basic classifier
        run_name = "iris_classifier_" + str(datetime.now().strftime("%d-%m-%y"))
        create_experiment(experiment_name, run_name, run_metrics, model, 'confusion_matrix.png')

    def predict(self):
        url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
        data = self.load_data(url)
        # data.head()
        target_column = 'class'
        X_train, X_test, y_train, y_test = self.train_test_split(data, target_column)
        import mlflow
        logged_model = 'runs:/a9dedd2db0814ba380ff1080f7609dcf/model'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on a Pandas DataFrame.
        import pandas as pd
        prediction = loaded_model.predict(pd.DataFrame(X_test))
        print(prediction)

    # Adding an MLflow Model to the Model Registry

    # Method 1

    def create_exp_and_register_model(self, experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                                      roc_auc_plot_path=None, run_params=None):
        mlflow.set_tracking_uri("http://localhost:5000")
        # use above line if you want to use any database like sqlite as backend storage for model else comment this line
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            if not run_params is None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])

            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])

            if not confusion_matrix_path is None:
                mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')

            if not roc_auc_plot_path is None:
                mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")

            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2": "Randomized Search CV", "tag3": "Production"})
            mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier")

    def train_and_register_model(self):
        url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
        data = self.load_data(url)
        # data.head()
        target_column = 'class'
        X_train, X_test, y_train, y_test = self.train_test_split(data, target_column)

        model = self.training_basic_classifier(X_train, y_train)

        print(f"See the prediction outcome")
        y_pred = self.predict_on_test_data(model, X_test)
        print(y_pred)
        y_pred_prob = self.predict_prob_on_test_data(model, X_test)
        print(y_pred_prob)
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        experiment_name = "iris_classifier_method-1"  # + str(datetime.now().strftime("%d-%m-%y")) ##basic classifier
        run_name = "iris_classifier_method-1"  # +str(datetime.now().strftime("%d-%m-%y"))
        self.create_exp_and_register_model(experiment_name, run_name, run_metrics, model, 'confusion_matrix.png')

    # Method 2
    def registry_method_2(self, run_name):
        """
        The second way is to use the mlflow.register_model() method,
        after all your experiment runs complete and when you have decided which model is most suitable to add to the registry.
        For this method, you will need the run_id as part of the runs:URI argument.
        """
        with mlflow.start_run(run_name=run_name) as run:
            result = mlflow.register_model(
                "runs:/dff923c9e0924e8e968eaed4cab33ee9/model",
                "iris-classifier-2"
            )

    # Method 3
    def registry_method_3(self):
        """
        And finally, you can use the create_registered_model() to create a new registered model. If the model name exists,
        this method will throw an MlflowException because creating a new registered model requires a unique name.
        """
        client = mlflow.tracking.MlflowClient()
        client.create_registered_model("basic-classifier-method-3")

        # While the method above creates an empty registered model with no version associated,

    def fetch_model_version(self, model_name):
        import mlflow.pyfunc

        # model_name = "iris-classifier"
        model_name = model_name
        model_version = 1
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

        url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
        data = self.load_data(url)
        # data.head()
        target_column = 'class'
        X_train, X_test, y_train, y_test = self.train_test_split(data, target_column)

        y_pred = model.predict(X_test)
        print(y_pred)

        sklearn_model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        y_pred_prob = sklearn_model.predict_proba(X_test)
        print(y_pred_prob)

    def model_staging(self, model_name, version, stage):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

    def predict_using_production_model(self, model_name, stage):

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
        data = self.load_data(url)
        # data.head()
        target_column = 'class'
        X_train, X_test, y_train, y_test = self.train_test_split(data, target_column)

        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{stage}"
        )

        y_pred = model.predict(X_test)
        print(y_pred)

    def serve_model(self):
        import os
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        os.system('mlflow models serve --model-uri models:/iris-classifier/Production -p 1234 --no-conda')


if __name__ == '__main__':
    model_name = 'iris-classifier'
    version = 1
    stage = "Production"
    mlflow_server = MLFlowModelServer()
    # mlflow_server.train_iris()
    # mlflow_server.predict()
    # mlflow_server.train_and_register_model()
    # mlflow_server.fetch_model_version(model_name)
    # mlflow_server.model_staging(model_name, version, stage)
    # mlflow_server.predict_using_production_model(model_name, stage)
    mlflow_server.serve_model()
