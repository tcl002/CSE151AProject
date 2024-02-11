# CSE151AProject

## Abstract

In this study, we will leverage the dataset titled, “Predict students dropout and academic” from UC Irvine’s Machine Learning Repository, uploaded on December 12th, 2021. Our objective is to employ supervised learning, specifically utilizing various regression models and a neural network, to predict the success rates of students in higher education. The dataset has 36 potential features, which covers historical information such as age, gender, nationality, etc. Our approach involves training a model to predict the target variable (ie. success) based on these student profiles.

## Dataset Links

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation

## Group Members

Kate Jackson; k1jackson@ucsd.edu

Emily Jin; ejin@ucsd.edu

James Thai; jqthai@ucsd.edu

Shol Hafezi; shafezi@ucsd.edu

Tyler Lee; tcl002@ucsd.edu

Andrew Cen; acen@ucsd.edu

Ka Wing Yan; w2yan@ucsd.edu

Kim Lim; kdlim@ucsd.edu

## Data Preprocessing

### 1. Encode Categorical Variables
The dataset contains several categorical variables encoded as integers, such as 'Marital status', 'Application mode', 'Course', etc. These need to be properly encoded to ensure they are correctly interpreted by the model. OneHotEncoding is used for the target variable to convert it into binary columns, which can be applied similarly to other categorical features if needed.

### 2. Normalize Numerical Variables
Numerical variables, especially those on different scales, can significantly impact model performance. MinMaxScaler is applied to scale numerical features between 0 and 1, ensuring uniformity in scale without distorting differences in the ranges of values.

### 3. Handle Missing Data
Although the dataset has no missing values, it's essential to have a strategy for handling them if they were to appear in updated data. Options include imputation, removing rows with missing values, or using algorithms that can handle missing values.

### 4. Correlation Analysis
The correlation analysis identified relationships between variables. Highly correlated features may introduce multicollinearity, affecting model performance. Features with high correlation might be candidates for removal or combined through feature engineering to reduce redundancy.

### 5. Data Visualization
Visualization techniques, including histograms and heatmaps for the correlation matrix, were employed to understand distributions and relationships within the data. This step is crucial for identifying patterns, outliers, and potential biases in the dataset.

### 6. Splitting the Data
The dataset will be split into training and test sets to evaluate the performance of the model accurately. This ensures that the model is tested on unseen data, providing a reliable estimate of its performance.

### 7. Balance the Target Variable
The distribution of the target variable ('Target') should be checked. If imbalanced, techniques like SMOTE, undersampling, or oversampling can be used to ensure the model does not become biased towards the majority class.

## Notebook Link
[https://colab.research.google.com/drive/1O30J-aRLqy5FdP2zSYkbxSQPZl2MfoAO?authuser=3#scrollTo=S5sWIqLq0t-F](https://drive.google.com/file/d/1O30J-aRLqy5FdP2zSYkbxSQPZl2MfoAO/view?usp=sharing)

### How to Set Up Python Environment (Optional)

If you run a project not in Datahub, and you run the project in local machine or other environment and meet some problems, you may need to set up python environment.

We have a `requirements.txt` file to install python packages. You can install the package by running the following command in the terminal(make sure in the correct dictionary):

``` sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instruction on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

``` sh
conda create -n cse151a python=3.9.5
conda activate cse151a
pip install -r requirements.txt
```