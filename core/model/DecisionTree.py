import numpy as np
import pickle
import os

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

class model_DecisionTree():
    """Baseline model for the ML lab challenge
    """
    def __init__(self,*args,**kwargs):
        self.model = DecisionTreeClassifier()
        pass

    def extract_features(self, x_single):
        """Add additional feature extraction to the input data in an
        on-the-fly manner
        """
        # Compute mean and standard deviation of the features
        x_mean = np.mean(x_single, axis=0)
        x_std = np.std(x_single, axis=0)
        x_processed = np.concatenate([x_mean, x_std])


        return x_processed

    def train(self, x, y, x_val, y_val, metric_fct, *args, **kwargs):
        """train model
        """
        # Convert y and y_val to 1d-arrays instead of column
        y = np.squeeze(y)
        y_val = np.squeeze(y_val)

        # Define hyper-parameters
        param_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'min_samples_split': [0.5, 2, 3]}

        # Do a gridsearch over the hyper-parameters
        bScore, bCrit, bSplit, bMinSample = 0, '', '', 0
        for criterion in param_grid['criterion']:
            for splitter in param_grid['splitter']:
                for min_samples_split in param_grid['min_samples_split']:
                    print('param: ' + str(criterion) + str(splitter) + str(min_samples_split))
                    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
                    model.fit(x, y)
                    y_val_hat = model.predict(x_val)
                    score = metric_fct(y_val, y_val_hat)
                    if score > bScore:
                        bScore = score
                        bCrit = criterion
                        bSplit = splitter
                        bMinSample = min_samples_split
        # Learn the model on both the training and validation data given the optimal hyper-parameters
        print('Best Param: ' + str(criterion) + str(splitter) + str(min_samples_split))

        model = DecisionTreeClassifier(criterion=bCrit, splitter=bSplit, min_samples_split=bMinSample)
        model.fit(np.concatenate([x, x_val]), np.concatenate([y, y_val]))

        # Store the model
        self.model = model

        self.model = self.model.fit(np.concatenate([x, x_val]), np.concatenate([y, y_val]))

    def predict(self, x, *arg,**kwargs):
        """test model
        """
        y_val_hat = self.model.predict(x)
        return y_val_hat


    def save(self,filepath, *arg,**kwargs):
        """Save model
        """
        # Check if the filepath already exists
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Save the model
        pickle.dump(self.model, open(os.path.join(filepath, 'model.pickle'), "wb"))

    def load(self,filepath, *arg,**kwargs):
        """Load model
        """
        # Load the model
        with open(os.path.join(filepath, "model.pickle"), "rb") as f:
            model = pickle.load(f)

        # Store the model
        self.model = model

    def reset(self):
        """Reset the model parameters
        """
        raise NotImplementedError

    def exists(self, filepath):
        """Check if model exists
        """
        # Check if file exists
        if os.path.isfile(os.path.join(filepath, 'model.pickle')):
            return True
        else:
            return False

    def __call__(self,data, *arg,**kwargs):
        """for the test phase, you can do y = model(x) instead of y = model.test(x)
        """
        return self.test(data,*arg,**kwargs)