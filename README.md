# EE 559 Project

Kaggle Dataset: https://www.kaggle.com/datasets/ara001/laptop-prices-based-on-its-specifications/data

## Preprocessing and Feature Engineering

### Step 1: Importing Training Dataset and Encoding 
We imported the data from `laptop_data_train.csv` and split some features into multiple features. 

We split the CPU category into 3 features: Core Line Name (i.e. Intel Core i7), Core Model/Generation (i.e. 6820HK), Processor Clock Speed (i.e. 2.7 GHz).


### Step 2: Splitting features into multiple features
We determined how to encode. Our choices were one-hot encoding and label encoding. 
&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — One-Hot Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Preserves uniqueness (each category gets its own binary column, i.e. no ordinal relationship imposed), and it works well with nominal categorical data where there is no intrinsic order. 
&nbsp;&nbsp;&nbsp;&nbsp; Cons: Increases dimensionality, and adds sparse matrices which can consume more memory and computational resources.

&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — Label Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Reduced dimensionality (one column for labels) saves memory and computational resources, preserves order, and simplicity (straightforward to implement)
&nbsp;&nbsp;&nbsp;&nbsp; Cons: May inadvertently introduce ordinal relationships where none exist, thus leading to potentially biased models

The preference of one-hot encoding for nominal categorical data with no inherent order vs. the preference for label encoding for ordinal categorical data with a clear order led us to decide which encoding to use based on the characteristics of each feature. 
Feature 1 (x1): Company

