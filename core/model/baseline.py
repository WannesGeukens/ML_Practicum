import numpy as np
import pickle
import os
from sklearn.svm import SVC

class baseline():
    """Baseline model for the ML lab challenge
    """
    def __init__(self,*args,**kwargs):
        """Init model
        """
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
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'Kernel': ['linear']}

        # Do a gridsearch over the hyper-parameters
        bScore, bC = 0, 0
        for C in param_grid['C']:
            print('C: ' + str(C))
            svm = SVC(kernel=param_grid['Kernel'][0], C=C)
            svm.fit(x, y)
            y_val_hat = svm.predict(x_val)
            score = metric_fct(y_val, y_val_hat)
            if score>bScore:
                bC = C
                bScore = score

        # Learn the SVM on both the training and validation data given the optimal hyper-parameters
        print('bC:  ' + str(bC))

        svm = SVC(kernel=param_grid['Kernel'][0], C=bC)
        svm.fit(np.concatenate([x, x_val]), np.concatenate([y, y_val]))

        # Store the model
        self.model = svm

    def predict(self, x, *arg,**kwargs):
        """test model
        """
        # Do a prediction on the data
        y_val_hat = self.model.predict(x)

        # return estimates
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