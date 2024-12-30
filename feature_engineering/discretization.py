# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:55:30 2024

@author: xiaowen.shou
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# from warnings import warn


# ChiMerge method modeified from https://github.com/tatsumiw/ChiMerge/blob/master/ChiMerge.py
# TODO: add more constraits to the discretized result.

class ChiMerge():
    """
    supervised discretization using the ChiMerge method.
    
    
    Parameters
    ----------
    confidenceVal: number
        default=3.841, correspond to p=0.05 dof=1
    num_of_bins: int
        number of bins after discretize
    col: str
        the column to be performed
        
    """
    
    def __init__(self, col=None, bins=None, confidenceVal=3.841, num_of_bins=10):
        self.col = col
        self._dim = None
        self.confidenceVal = confidenceVal
        self.bins = bins
        self.num_of_bins = num_of_bins


    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        self._dim = X.shape[1]

        _, bins = self.chimerge(
            X_in=X,
            y=y,
            confidenceVal=self.confidenceVal,
            col=self.col,
            num_of_bins=self.num_of_bins
        )
        self.bins = bins
        return self
    
    
    def transform(self, X):
            """Perform the transformation to new data.
            Will use the tree model and the column list to discretize the
            column.
            Parameters
            ----------
            X : array-like, shape = [n_samples, n_features]
            Returns
            -------
            X : new dataframe with discretized new column.
            """
    
            if self._dim is None:
                raise ValueError('Must train encoder before it can be used to transform data.')
    
            #  make sure that it is the right size
            if X.shape[1] != self._dim:
                raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
    
            X, _ = self.chimerge(
                X_in=X,
                col=self.col,
                bins=self.bins
            )
    
            return X 

    def chimerge(self, X_in, y=None, confidenceVal=None, num_of_bins=None, col=None, bins=None):
        """
        discretize a variable using ChiMerge

        """

        X = X_in.copy(deep=True)

        if bins is not None:  # transform
            try:
                X[col+'_chimerge'] = pd.cut(X[col],bins=bins,include_lowest=True)
            except Exception as e:
                print(e)
       
        else: # fit
            try:               
                # create an array which save the num of 0/1 samples of the column to be chimerge
                total_num = X.groupby([col])[y].count()
                total_num = pd.DataFrame({'total_num': total_num}) 
                positive_class = X.groupby([col])[y].sum()
                positive_class = pd.DataFrame({'positive_class': positive_class}) 
                regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,how='inner')  
                regroup.reset_index(inplace=True)
                regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  
                regroup = regroup.drop('total_num', axis=1)
                np_regroup = np.array(regroup)  
                # merge interval that have 0 pos/neg samples
                i = 0
                while (i <= np_regroup.shape[0] - 2):
                    if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                        np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # pos
                        np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # neg
                        np_regroup[i, 0] = np_regroup[i + 1, 0]
                        np_regroup = np.delete(np_regroup, i + 1, 0)
                        i = i - 1
                    i = i + 1
                # calculate chi for neighboring intervals
                # ∑[(yA-yB)²/yB]
                chi_table = np.array([])
                for i in np.arange(np_regroup.shape[0] - 1):
                    chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
                      * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
                      ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                      np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
                    chi_table = np.append(chi_table, chi)
                # merge intervals that have closing chi
                while (1):
                    if (len(chi_table) <= (num_of_bins - 1) and min(chi_table) >= confidenceVal):
                        break
                    chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  
                    np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
                    np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
                    np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
                    np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)
        
                    if (chi_min_index == np_regroup.shape[0] - 1): 
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                   ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table = np.delete(chi_table, chi_min_index, axis=0)
        
                    else:
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                   * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                   ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                                   * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                               ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                        chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
                result_data = pd.DataFrame()
                result_data['variable'] = [col] * np_regroup.shape[0]
                bins = []
                tmp = []
                for i in np.arange(np_regroup.shape[0]):
                    if i == 0:
                        y = '-inf' + ',' + str(np_regroup[i, 0])
                        #x = np_regroup[i, 0]
                        #list_temp.append(x)
                    elif i == np_regroup.shape[0] - 1:
                        y = str(np_regroup[i - 1, 0]) + '+'
                        #x = 100000000.
                        #list_temp.append(x)
                    else:
                        y = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
                        #x = np_regroup[i, 0]
                        #list_temp.append(x)
                    bins.append(np_regroup[i - 1, 0])
                    tmp.append(y)
                
                #list_temp.append(df[variable].max()+0.1)
                bins.append(X[col].min()-0.1)
                
                result_data['interval'] = tmp  
                result_data['flag_0'] = np_regroup[:, 2] 
                result_data['flag_1'] = np_regroup[:, 1]  
                bins.sort(reverse=False)
                print('Interval for variable %s' % col)
                print(result_data)
                
            except Exception as e:
                print(e)
        
        return X, bins
        
        
        
        

class DiscretizeByDecisionTree():
    """
    Discretisation with Decision Trees consists of using a decision tree 
    to identify the optimal splitting points that would determine the bins 
    or contiguous intervals:  
        
    1.train a decision tree of limited depth (2, 3 or 4) using the variable 
    we want to discretise to predict the target.
    2.the original variable values are then replaced by the 
    probability returned by the tree.

    Parameters
    ----------
    col: str
      column to discretise
    max_depth: int or list of int
      max depth of the tree. Can be an int or a list of int we want the tree model to search 
      for the optimal depth.
    
    """

    def __init__(self, col=None, max_depth=None, tree_model=None):
        self.col = col
        self._dim = None
        self.max_depth = max_depth
        self.tree_model = tree_model


    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        self._dim = X.shape[1]

        _, tree = self.discretize(
            X_in=X,
            y=y,
            max_depth=self.max_depth,
            col=self.col,
            tree_model=self.tree_model
        )
        self.tree_model = tree
        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.
        Will use the tree model and the column list to discretize the
        column.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        X : new dataframe with discretized new column.
        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        #  make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        X, _ = self.discretize(
            X_in=X,
            col=self.col,
            tree_model=self.tree_model
        )

        return X 


    def discretize(self, X_in, y=None, max_depth=None, tree_model=None, col=None):
        """
        discretize a variable using DecisionTreeClassifier

        """

        X = X_in.copy(deep=True)

        if tree_model is not None:  # transform
            X[col+'_tree_discret'] = tree_model.predict_proba(X[col].to_frame())[:,1]

        else: # fit
            if isinstance(max_depth,int):
                tree_model = DecisionTreeClassifier(max_depth=max_depth)
                tree_model.fit(X[col].to_frame(), y)
                # X[col+'_tree_discret'] = tree_model.predict_proba(X[col].to_frame())[:,1]
                #print(x.tree_discret.unique())
#                bins = pd.concat( [X.groupby([col+'_tree_discret'])[col].min(),
#                                  X.groupby([col+'_tree_discret'])[col].max()], axis=1)
#                print('bins:')            
#                print(bins)
            
            elif isinstance(max_depth, (list, np.ndarray)):
                # Multiple depths provided, search for the best
                score_ls = []  # Store the mean ROC AUC scores
                score_std_ls = []  # Store the standard deviation of ROC AUC scores

                for tree_depth in max_depth:
                    temp_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
                    scores = cross_val_score(temp_model, X[col].to_frame(), y, cv=3, scoring='roc_auc')
                    score_ls.append(np.mean(scores))
                    score_std_ls.append(np.std(scores))

                # Collect results into a DataFrame
                temp = pd.DataFrame({
                    'depth': max_depth,
                    'roc_auc_mean': score_ls,
                    'roc_auc_std': score_std_ls
                })
                print("Result ROC-AUC for each depth:")
                print(temp)

                # Find the optimal depth
                max_roc = temp['roc_auc_mean'].max()
                optimal_depth = temp.loc[temp['roc_auc_mean'] == max_roc, 'depth'].values[0]
                print("Optimal depth:", optimal_depth)

                # Train the model with the optimal depth
                tree_model = DecisionTreeClassifier(max_depth=int(optimal_depth), random_state=42)
                tree_model.fit(X[col].to_frame(), y)

            else:
                raise ValueError("max_depth of a tree must be an integer or a list of integers.")

        return X, tree_model


