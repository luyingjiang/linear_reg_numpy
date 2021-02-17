'''
Linear regression

Luying Jiang

Main file for linear regression and model selection.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import util


class DataSet(object):
    '''
    Class for representing a data set.
    '''

    def __init__(self, dir_path):
        '''
        Constructor
        Inputs:
            dir_path: (string) path to the directory that contains the
              file
        '''
        self.dir_path = dir_path
        params_dict = util.load_json_file(self.dir_path, 'parameters.json')
        self.label, data = util.load_numpy_array(self.dir_path, 'data.csv')
        self.pred_vars = params_dict['predictor_vars']
        self.dep_var = params_dict['dependent_var']
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data[:,self.pred_vars], data[:,self.dep_var], 
                            train_size = params_dict['training_fraction'], 
                            random_state = params_dict['seed'])
        

class Model(object):
    '''
    Class for representing a model.
    '''

    def __init__(self, dataset, pred_vars):
        '''
        Construct a data structure to hold the model.
        Inputs:
            dataset: an dataset instance
            pred_vars: a list of the indices for the columns (of the
              original data array) used in the model.
        '''
        self.dataset = dataset
        self.pred_vars = pred_vars
        self.dep_var = self.dataset.dep_var
        self.X_train = util.prepend_ones_column(dataset.X_train[:,pred_vars])
        self.X_testing = \
            util.prepend_ones_column(dataset.X_test[:,pred_vars])
        self.beta = util.linear_regression(self.X_train, dataset.y_train)
        self.R2 = self.cal_R2(self.X_train, self.dataset.y_train)
        self.adj_R2 = self.R2 - (1 - self.R2) * len(self.pred_vars) / \
            (len(dataset.X_train) - len(self.pred_vars) - 1)
        

    def __repr__(self):
        '''
        Format model as a string.
        '''
        string = "{} ~ {}".format(self.dataset.label[self.dep_var], \
            np.round(self.beta[0], decimals = 6))

        for i in range(len(self.beta) - 1):
            string += " + {} * {}".format(np.round(self.beta[i + 1], \
                decimals = 6), self.dataset.label[self.pred_vars[i]])

        return string
        

    def cal_R2(self, X, y):
        '''
        Computes the R-square value for the model

        Inputs:
            X: an N×(K+1) matrix where each row is one sample unit. 
               (The first column of this matrix is all ones.)
            y: a column vector of observations of the dependent variable.

        Returns:
            (float) An R2 value
        '''
        y_bar = y.mean()
        y_hat = util.apply_beta(self.beta, X)
        numerator = sum((y - y_hat) ** 2)
        denominator = sum((y - y_bar) ** 2)
        
        return (1 - numerator / denominator)


def compute_single_var_models(dataset):
    '''
    Computes all the single-variable models for a dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        List of Model objects, each representing a single-variable model
    '''
    return [Model(dataset, [pred_vars]) for pred_vars in dataset.pred_vars]
    

def compute_all_vars_model(dataset):
    '''
    Computes a model that uses all the predictor variables in the dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object that uses all the predictor variables
    '''
    return Model(dataset, dataset.pred_vars)


def compute_best_pair(dataset):
    '''
    Find the bivariate model with the best R2 value

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object for the best bivariate model
    '''
    max_val = 0
    for i in range(len(dataset.pred_vars)):
        for j in range(i + 1, len(dataset.pred_vars)):
            model = Model(dataset, [i,j])
            if model.R2 > max_val:
                max_val = model.R2
                best_model = model

    return best_model


def backward_elimination(dataset):
    '''
    Given a dataset with P predictor variables, uses backward elimination to
    select models for every value of K between 1 and P.

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A list (of length P) of Model objects. The first element is the
        model where K=1, the second element is the model where K=2, and so on.
    '''
    lst = []
    lst.append(Model(dataset,dataset.pred_vars))
    pred_vars = dataset.pred_vars

    for K in range(len(dataset.pred_vars) - 1):
        max_val = 0
        for i in range(len(pred_vars)):
            pred = pred_vars[:i] + pred_vars[i+1:]
            model = Model(dataset, pred)
            if model.R2 > max_val:
                max_val = model.R2
                best_model = model
                best_pred = pred
        lst.append(best_model)
        pred_vars = best_pred

    return lst[::-1]


def choose_best_model(dataset):
    '''
    Given a dataset, choose the best model produced
    by backwards elimination (i.e., the model with the highest
    adjusted R2)

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object
    '''
    lst = backward_elimination(dataset)
    max_val = 0
    for model in lst:
        if model.adj_R2 > max_val:
            max_val = model.adj_R2
            best_model = model
            
    return best_model


def validate_model(dataset, model):
    '''
    Given a dataset and a model trained on the training data,
    compute the R2 of applying that model to the testing data.

    Inputs:
        dataset: (DataSet object) a dataset
        model: (Model object) A model that must have been trained
           on the dataset's training data.

    Returns:
        (float) An R2 value
    '''
    return model.cal_R2(model.X_testing, dataset.y_test)