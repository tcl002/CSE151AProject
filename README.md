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


## Notebook Link
[https://colab.research.google.com/drive/1O30J-aRLqy5FdP2zSYkbxSQPZl2MfoAO?authuser=3#scrollTo=S5sWIqLq0t-F](https://drive.google.com/file/d/1O30J-aRLqy5FdP2zSYkbxSQPZl2MfoAO/view?usp=sharing)

### How to Set Up Python Environment (Optional)

If you run a project not in Datahub, and you run the project in local machine or other environment and meet some problems, you may need to set up python environment.

We have a `requirements.txt` file to install python packages. You can install the package by running the following command in the terminal(make sure in the correct dictionary):

```sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instruction on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

```sh
conda create -n cse151a python=3.9.5  
conda activate cse151a
pip install -r requirements.txt
```
