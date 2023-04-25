import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


class MlflowLiveDemo:
    def load_data(self, path):
        data = pd.read_csv(path)
        return data

    def data_cleaning(self, data):
        print("na values available in data \n")
        print(data.isna().sum())
        data = data.dropna()
        print("after droping na values \n")
        print(data.isna().sum())
        return data

    def preprocessing(self, data):
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

    def train_test_split(self, final_data):
        from sklearn.model_selection import train_test_split
        X = final_data.loc[:, final_data.columns != 'y']
        y = final_data.loc[:, final_data.columns == 'y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)
        return X_train, X_test, y_train, y_test

    def over_sampling_target_class(self, X_train, y_train):
        ### Over-sampling using SMOTE
        from imblearn.over_sampling import SMOTE
        os = SMOTE(random_state=0)

        columns = X_train.columns
        os_data_X, os_data_y = os.fit_resample(X_train, y_train)

        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
        # we can Check the numbers of our data
        print("length of oversampled data is ", len(os_data_X))
        print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['y'] == 0]))
        print("Number of subscription", len(os_data_y[os_data_y['y'] == 1]))
        print("Proportion of no subscription data in oversampled data is ",
              len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))
        print("Proportion of subscription data in oversampled data is ",
              len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X))

        X_train = os_data_X
        y_train = os_data_y['y']

        return X_train, y_train

    def training_basic_classifier(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=101)
        model.fit(X_train, y_train)

        return model

    def predict_on_test_data(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def predict_prob_on_test_data(self, model, X_test):
        y_pred = model.predict_proba(X_test)
        return y_pred

    def get_metrics(self, y_true, y_pred, y_pred_prob):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        entropy = log_loss(y_true, y_pred_prob)
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2),
                'entropy': round(entropy, 2)}

    def create_roc_auc_plot(self, clf, X_data, y_data):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        y_pred_proba = clf.predict_proba(X_data)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(X_data, y_data)
        auc = metrics.roc_auc_score(X_data, y_pred_proba)

        # create ROC curve
        plt.plot(fpr, tpr, label="AUC=" + str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()
        plt.savefig('roc_auc_curve.png')

    def create_confusion_matrix_plot(self, clf, X_test, y_test):
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = confusion_matrix(X_test, y_test, labels=clf.classes_)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        # disp.plot()
        plt.savefig('confusion_matrix.png')

    def hyper_parameter_tuning(self, X_train, y_train):
        # define random parameters grid
        n_estimators = [5, 21, 51, 101]  # number of trees in the random forest
        max_features = ['auto', 'sqrt']  # number of features in consideration at every split
        max_depth = [int(x) for x in
                     np.linspace(10, 120, num=12)]  # maximum number of levels allowed in each decision tree
        min_samples_split = [2, 6, 10]  # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
        bootstrap = [True, False]  # method used to sample data points

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap
                       }

        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
        model_tuning = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid,
                                          n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1)
        model_tuning.fit(X_train, y_train)

        print('Random grid: ', random_grid, '\n')
        # print the best parameters
        print('Best Parameters: ', model_tuning.best_params_, ' \n')

        best_params = model_tuning.best_params_

        n_estimators = best_params['n_estimators']
        min_samples_split = best_params['min_samples_split']
        min_samples_leaf = best_params['min_samples_leaf']
        max_features = best_params['max_features']
        max_depth = best_params['max_depth']
        bootstrap = best_params['bootstrap']

        model_tuned = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf, max_features=max_features,
                                             max_depth=max_depth, bootstrap=bootstrap)
        model_tuned.fit(X_train, y_train)
        return model_tuned, best_params

    ## Function to create an experiment in MLFlow and log parameters, metrics and artifacts files like images etc.

    def create_experiment(self, experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                          roc_auc_plot_path=None, run_params=None):
        import mlflow
        # mlflow.set_tracking_uri("http://localhost:5000") #uncomment this line if you want to use any database like sqlite as backend storage for model
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():

            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])

            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])

            mlflow.sklearn.log_model(model, "model")

            if not confusion_matrix_path == None:
                mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')

            if not roc_auc_plot_path == None:
                mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")

            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2": "Randomized Search CV", "tag3": "Production"})

        print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))

    def train_base_classifier(self, data_path):
        data = self.load_data(data_path)
        cleaned_data = self.data_cleaning(data)
        final_data = self.preprocessing(cleaned_data)
        X_train, X_test, y_train, y_test = self.train_test_split(final_data)
        X_train, y_train = self.over_sampling_target_class(X_train, y_train)

        model = self.training_basic_classifier(X_train, y_train)
        y_pred = self.predict_on_test_data(model, X_test)
        y_pred_prob = self.predict_prob_on_test_data(model, X_test)  # model.predict_proba(X_test)

        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        # self.create_roc_auc_plot(model, X_test, y_test)
        self.create_confusion_matrix_plot(model, X_test, y_test)

        experiment_name = "basic_classifier"  ##basic classifier
        run_name = "term_deposit"

        # self.create_experiment(experiment_name, run_name, run_metrics, model, 'confusion_matrix.png', 'roc_auc_curve.png')
        self.create_experiment(experiment_name, run_name, run_metrics, model)

    # def mlflow_base_clf_work(self):
    #     experiment_name = "basic_classifier"  ##basic classifier
    #     run_name = "term_deposit"
    #     run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)

    ## Create another experiment after tuning hyperparameters and log the best set of parameters for which model gives the optimal performance

    def train_optimize_clf(self, data_path):
        experiment_name = "optimized model"
        run_name = "Random_Search_CV_Tuned_Model"
        data = self.load_data(data_path)
        cleaned_data = self.data_cleaning(data)
        final_data = self.preprocessing(cleaned_data)
        X_train, X_test, y_train, y_test = self.train_test_split(final_data)
        X_train, y_train = self.over_sampling_target_class(X_train, y_train)
        model_tuned, best_params = self.hyper_parameter_tuning(X_train, y_train)
        run_params = best_params

        y_pred = self.predict_on_test_data(model_tuned, X_test)  # will return the predicted class
        y_pred_prob = self.predict_prob_on_test_data(model_tuned, X_test)  # model.predict_proba(X_test)
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        for param in run_params:
            print(param, run_params[param])
        self.create_experiment(experiment_name, run_name, run_metrics, model_tuned, run_params=run_params)


if __name__ == '__main__':
    data_path = 'banking.csv'
    mlflow_live_demo = MlflowLiveDemo()
    # mlflow_live_demo.train_base_classifier(data_path)
    mlflow_live_demo.train_optimize_clf(data_path)
