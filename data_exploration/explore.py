# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:19:03 2024

@author: xiaowen.shou
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-v0_8-colorblind')

# 2024.12.11 Created by Xiaowen.Shou


def get_dtypes(data, drop_col=[]):
    """Return the dtypes for each column of a pandas Dataframe
    Parameters
    ----------
    data: pandas Dataframe
    drop_col: columns to omit in a list

    Returns
    -------
    str_var_list, num_var_list, all_var_list
    """
    name_of_col = list(data.columns)
    num_var_list = []
    str_var_list = []
    all_var_list = []

    str_var_list = name_of_col.copy()
    for var in name_of_col:
        # check if column belongs to numeric type
        if (data[var].dtype in (int, np.int64, np.uint, np.int32, float, np.float64, np.float32, np.double)):
            str_var_list.remove(var)
            num_var_list.append(var)
    
    # drop the omit column from list
    for var in drop_col:
        if var in str_var_list:
            str_var_list.remove(var)
        if var in num_var_list:
            num_var_list.remove(var)

    all_var_list.extend(str_var_list)
    all_var_list.extend(num_var_list)
    
    return str_var_list, num_var_list, all_var_list


def describe(data, output_path=None):
    """output the general description of a pandas Dataframe into a csv file."""
    result = data.describe(include='all')
    if output_path is not None:
        output = os.path.join(output_path,'describe.csv')
        result.to_csv(output)
        print('result saved at:', str(output))
    return result


def discrete_var_barplot(x, y, data, output_path=None):
    """draw the barplot of a discrete variable x against y (target variable).
    By default the bar shows the mean value of y.

    Parameters
    ----------

    Returns
    -------
    figure saved as PNG
    """

    plt.figure(figsize=(15,10))
    sns.barplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Barplot_'+str(x)+'_'+str(y)+'.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def discrete_var_countplot(x, data, output_path=None):
    """draw the countplot of a discrete variable x.

    Parameters
    ----------

    Returns
    -------
    figure saved as PNG
    """

    plt.figure(figsize=(15,10))
    sns.countplot(x=x, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Countplot_'+str(x)+'.png')
        plt.savefig(output)
        print('Image saved at', str(output))
        
def discrete_var_boxplot(x, y, data, output_path=None):
    """draw the boxplot of a discrete variable x against y (target variable).

    Parameters
    ----------

    Returns
    -------
    figure saved as PNG
    """

    plt.figure(figsize=(15,10))
    sns.boxplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Boxplot_'+str(x)+'_'+str(y)+'.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def continuous_var_histplot(x, output_path=None, bins=10):
    """draw the histplot of a continuous variable x.

    Parameters
    ----------

    Returns
    -------
    figure saved as PNG
    """

    plt.figure(figsize=(15,10))
    sns.histplot(x=x, kde=False, bins=bins)
    if output_path is not None:
        output=os.path.join(output_path,'Distplot_'+str(x.name)+'.png')
        plt.savefig(output)
        print('Image saved at', str(output))


# 2024.12.11 Created by Xiaowen.Shou

def scatter_plot(x,y,data,output_path=None):
    """draw the scatter-plot of two variables.
    Parameters
    ----------
    
    Returns
    -------
    figure save as PNG
    """    
    
    plt.figure(figsize=(15,10))
    sns.scatterplot(x=x,y=y,data=data)
    if output_path is not None:
        output = os.path.join(output_path,'Scatter_plot_'+str(x.name)+'_'+str(y.name)+'.png')
        plt.savefig(output)
        print('Image saved at',str(output))       
        
    
def correlation_plot(data,output_path=None):
    """draw the correlation plot between variables.
    Parameters
    ----------

    Returns
    -------
    figure save as PNG
    """    
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])  # Automatically filters numeric columns

    # Check if numeric_data has at least two numeric columns
    if numeric_data.shape[1] < 2:
        raise ValueError("Not enough numeric columns for a correlation plot.")

    # Calculate the correlation matrix
    corrmat = numeric_data.corr()
   
    # Create a heatmap
    fig, ax = plt.subplots()
    fig.set_size_inches(11,11)
    sns.heatmap(corrmat,cmap="YlGnBu",linewidths=.5,annot=True)

    # Save the figure if output_path is provided
    if output_path is not None:
        output = os.path.join(output_path,'Corr_plot'+'.png')
        plt.savefig(output)
        print('Image saved at',str(output))  
    
    
def heatmap(data,output_path=None,fmt='d'):
    """draw the heatmap between 2 variables.
    Parameters
    ----------

    Returns
    -------
    figure save as PNG
    """    
    
    fig, ax = plt.subplots()
    fig.set_size_inches(11,11)
    sns.heatmap(data,cmap="YlGnBu",linewidths=.5,annot=True,fmt=fmt)
    if output_path is not None:
        output = os.path.join(output_path,'Heatmap'+'.png')
        plt.savefig(output)
        print('Image saved at',str(output)) 