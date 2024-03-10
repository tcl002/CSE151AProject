# CSE151AProject

## Introduction

In this study, we will leverage the dataset titled, “Predict students dropout and academic” from UC Irvine’s Machine Learning Repository, uploaded on December 12th, 2021. Our objective is to employ supervised learning, specifically utilizing a logistic regression model, a deep neural network, and a support vector machine, to predict the dropout rates of students in higher education. The dataset has 36 potential features, which covers historical information such as age, gender, nationality, etc. Our approach involves training a model to predict the target variable (ie. dropout) based on these student profiles.

We chose this dataset because as college students we are interested in being able to predict other students' trajectories in higher education. Specifically, we are interested in predicting whether a student will graduate based on their profile information. We find it cool to consider what variables have predictive power in determining whether a student will succeed in academics. Knowing whether grades, family background, financial background, or other factors affect a students chance of graduating would allow us to reflect on our own circumstances and those of our peers. Aditionally, this model is important because it could be utilized by higher education institutions to predict which students are likely to dropout and possibly intervene beforehand to get them the support they need to graduate. This is especially important giving that national collage dropout rates are at 40% and on the rise (![source](https://research.com/universities-colleges/college-dropout-rates#2)). Our model could potentially become a source of information which allows universities to focus their resources on those who need them most.

### Dataset Links

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

### Notebook Link
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

# Methods

### Data Exploration

### Data Preprocessing

#### 1. Encode Categorical Variables
The dataset contains several categorical variables encoded as integers, such as 'Marital status', 'Application mode', 'Course', etc. These need to be properly encoded to ensure they are correctly interpreted by the model. OneHotEncoding is used for the target variable to convert it into binary columns, which can be applied similarly to other categorical features if needed.

#### 2. Normalize Numerical Variables
Numerical variables, especially those on different scales, can significantly impact model performance. MinMaxScaler is applied to scale numerical features between 0 and 1, ensuring uniformity in scale without distorting differences in the ranges of values.

#### 3. Handle Missing Data
Although the dataset has no missing values, it's essential to have a strategy for handling them if they were to appear in updated data. Options include imputation, removing rows with missing values, or using algorithms that can handle missing values.

#### 4. Correlation Analysis
The correlation analysis identified relationships between variables. Highly correlated features may introduce multicollinearity, affecting model performance. Features with high correlation might be candidates for removal or combined through feature engineering to reduce redundancy.

#### 5. Data Visualization
Visualization techniques, including histograms and heatmaps for the correlation matrix, were employed to understand distributions and relationships within the data. This step is crucial for identifying patterns, outliers, and potential biases in the dataset.

#### 6. Splitting the Data
The dataset will be split into training and test sets to evaluate the performance of the model accurately. This ensures that the model is tested on unseen data, providing a reliable estimate of its performance.

#### 7. Balance the Target Variable
The distribution of the target variable ('Target') should be checked. If imbalanced, techniques like SMOTE, undersampling, or oversampling can be used to ensure the model does not become biased towards the majority class.

## Model 1: Logistic Regression Model

### Evaluation of Training vs. Testing Error
Out best logistic regression model achieved an accuracy of 0.89 (precision of 0.90, recall of 0.73, loss of 3.52) on the training data and an accuracy of 0.86 (precision of 0.87, recall of 0.72, loss of 5.07) on the testing data. There is a minimal difference between the model's performance on the training and testing data which indicates that we did not overfit our model (which is also reflected in out validation data). Overall, we think that the logistic regression model performed well but increasing the model complexity (as in a NN) may improve the accuracy. 

### Model Fitting
First we split out data into numerical, categorical (multiclass), categorical (binary), and one-hot-encoded categorical (multiclass) data. We determined that the model performed best on numerical data (accuracy of 0.85), binary categorical data (accuracy of 0.77), and one-hot-encoded multiclass categoical data (accuracy of 0.755). Next, we fitted our model to the optimal number of features. We found that the optimal number of features was 35 (mostly numerical and binary features) which gave us a validation accuracy of approximately 0.89.

### Model Selection
1. Our first model (for Milestone 4) following our logistic regression model is going to be a deep neural net (DNN). Our reasoning for this is because our target class is categorical with three separate categories to determine between. A neural net would allow us to create a model that outputs predictions for all three categories.
2. Our second model is going to be a support vector machine (SVM). Our plan for this is to combine the "Enrolled" and "Graduate" categories together so that we can have a categorical class that can be represented in a binary fashion. This way, we can use the SVM and compare it with our original logistic regression model to compare and contrast the models.

### Milestone 3 Conclusion
Based on the analysis of our logistic regression model trained on the dataset, it appears to perform reasonably well, achieving satisfactory accuracy and loss metrics on both the training and validation data. The model effectively leverages carefully selected features to make accurate predictions regarding whether students drop out at the end of the semester or continue their enrollment and/or graduate.

However, while the model demonstrates promising performance, there are still opportunities for improvement. One such avenue is the exploration of regularization techniques. Regularization can be beneficial in mitigating the risk of overfitting, particularly in scenarios where there's uncertainty about the relevance of certain features or the presence of noise in the data. By introducing regularization, such as L1 (lasso) or L2 (ridge) regularization, we can encourage the model to learn simpler patterns that generalize better to unseen data, thus potentially enhancing its overall performance and robustness.

## Model 2: Deep Neural Network

### Evalutation of Data, Labels, and Loss Function

In terms of out data, we believe the preprocessing from the previous model is still ideal. We think it is appropriate to one-hot encode categorical variables becuase they should be treated as independent features (as opposed to one label-encoded feature). Additionally, we think some form of scaling is needed for the numerical data, so we decided to use the min-max scaled data which is been squeezed between the range from 0 to 1 to improve out model's performance. We also will continue to use the features which were produced the best result in our previous model because we predict these features have the highest correlation with a student's dropout rate. 

In terms of the loss function, our previous model, logistic regression, uses log loss. In this model we decided to use binary crossentropy loss since we are attempting a binary classification (dropout or not dropout). During hyperparameter tuning we also experimented with using mean-squared error, but we found this loss function to be less effective than binary crossentropy.

### Evaluation of Training vs. Testing Error

For out best and final neural network, we achieved a training accuracy of 0.88, validation accuracy of 0.87, and a testing accuracy of 0.86 (loss of 0.31, 0.33, and 0.35 respectively). By performing K-fold cross-validation, we were able to ensure that there is a minimal difference between the training, validation, and testing errors which indicates that out model is not overfitting. In comparison to the previous model, the neural network achieved effectively identical training and testing accuracy to out logistic regression model. This indicates that both models are effective at using the selected features to classify students as dropouts or non-dropouts. 

### Model Fitting

### Hyperparameter Tuning, K-Fold Cross Validation, and Feature Expansion
To refine the performance of our model, we performed hyperparameter tuning using `keras_tuner.RandomSearch.` The key hyperparameters that were tuned include the number of units in each layer (32 to  512, with a step size of 32), the activation function (sigmoid, relu, tanh, and softmax), the optimizer (SGD and RMSprop), and the learning rate (1e-4 to 1e-2). We found that the most effective model parameters were 224 units in each hidden layer using tanh activation functions. We also found that an SGD optimizer with a learning rate of approximately 0.002 and binary crossentropy loss was most effective. This model achieved a training accuracy of 0.96, a validation accuracy of 0.83, and a testing accuracy of 0.84.

However because the large discrepency between the high training accuracy and the low validation / testing accuracy, we decided that our model was not generalizable enough and was most likely overfitting the data. Therefore, we employed a 10-fold cross-validation technique, which produced our best model. This model has a training accuracy of 0.88, validation accuracy of 0.87, and a testing accuracy of 0.86 (loss of 0.31, 0.33, and 0.35 respectively).

### Model Selection
Our third and final model (for Milestone 5) is still going to be a support vector machine (SVM). We believe an SVM will do a good jop at splitting the data into two classes (dropout and non-dropout) and we would like to experiment with different kernal functions to see if we can achieve a higher accuracy.

### Milestone 4 Conclusion
In the project, we developed two predictive models: a logistic regression model and a deep neural network, and the purpose is predict student dropout rates. The logistic regression model, our first approach, provided a baseline with notable strengths in interpretability and speed.The second model, a deep neural network, represented a more advanced approach, leveraging a sequential architecture with multiple dense layers. This model showed a marked improvement, particularly in handling non-linear patterns, with validation and test accuracy scores improving to 87% and 86%, respectively. This suggests that the deep neural network could capture complex interactions between student features more effectively than logistic regression.To further enhance the neural network model, we could experiment with additional layers, different activation functions, or more sophisticated regularization techniques to combat overfitting, as indicated by the gap between training and validation performance. Incorporating dropout layers or exploring different optimization algorithms could also yield performance gains. Comparatively, the deep neural network performed better than the logistic regression model in terms of handling complex data structures, as evidenced by higher accuracy and more nuanced pattern recognition in the dropout predictions. However, the logistic regression model maintained its value through ease of use and interpretation, crucial for stakeholders understanding contributing factors to student dropout.

## Model 3: Support Vector Machine

## Discussion

## Conclusion

## Collaboration
