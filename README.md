# CSE151AProject

## Introduction

In this study, we will leverage the dataset titled, “Predict Students' Dropout and Academic Success” from UC Irvine’s Machine Learning Repository, uploaded on December 12th, 2021. Our objective is to employ supervised learning, specifically utilizing a logistic regression model, a deep neural network, and a support vector machine, to predict the dropout rates of students in higher education. The dataset has 36 potential features, which cover historical information such as age, gender, nationality, etc. Our approach involves training a model to predict the target variable (ie. dropout) based on these student profiles.

We chose this dataset because as college students we are interested in being able to predict other students' trajectories in higher education. Specifically, we are interested in predicting whether a student will graduate based on their profile information. We find it cool to consider what variables have predictive power in determining whether a student will succeed in academics. Knowing whether grades, family background, financial background, or other factors affect a student's chance of graduating would allow us to reflect on our circumstances and those of our peers. Additionally, this model is important because it could be utilized by higher education institutions to predict which students are likely to drop out and possibly intervene beforehand to get them the support they need to graduate. This is especially important given that national college dropout rates are at 40% and on the rise (![source](https://research.com/universities-colleges/college-dropout-rates#2)). Our model could potentially become a source of information that allows universities to focus their resources on those who need them most.

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

If you run a project not in Datahub, and you run the project in a local machine or another environment and encounter some problems, you may need to set up Python environment.

We have a `requirements.txt` file to install python packages. You can install the package by running the following command in the terminal(make sure in the correct dictionary):

``` sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instructions on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

``` sh
conda create -n cse151a python=3.9.5
conda activate cse151a
pip install -r requirements.txt
```

## Methods

### Data Exploration
To prepare for the predictive modeling process, we conducted data exploration as a preliminary step to better understand the dataset's characteristics. 

* **Number of Instances:** The dataset contains 4424 instances, or students, which is indicative of a robust sample size.
* **Number of Features:** The dataset contains 36 features that encompass a range of information types from binary values to complex numerical scores.

As a part of our data exploration, we performed correlation analysis to identify any strong relationships between features, and the results of this analysis can be visualized in the heatmap below: 

![Heatmap](https://github.com/tcl002/CSE151AProject/blob/4756c21176c69aaf3b9651e4701874c2f99e08b6/graphs/Data%20Visualization/heatmap.png)

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
## Logistic Regression Model Performance

| Data Type | Accuracy | Precision | Recall | Loss  |
|-----------|----------|-----------|--------|-------|
| Training  | 0.89     | 0.90      | 0.73   | 3.52  |
| Testing   | 0.86     | 0.87      | 0.72   | 5.07  |

Out best logistic regression model achieved an accuracy of 0.89 (precision of 0.90, recall of 0.73, loss of 3.52) on the training data and an accuracy of 0.86 (precision of 0.87, recall of 0.72, loss of 5.07) on the testing data. There is a minimal difference between the model's performance on the training and testing data which indicates that we did not overfit our model (which is also reflected in our validation data). Overall, we think that the logistic regression model performed well but increasing the model complexity (as in a NN) may improve the accuracy. 



### Model Fitting
We categorized our dataset into four types: numerical, categorical (multiclass), categorical (binary), and one-hot-encoded categorical (multiclass). Here's how the logistic regression model performed across different data types:

| Data Type                                 | Accuracy |
|-------------------------------------------|----------|
| Numerical                                 | 0.85     |
| Categorical (Binary)                      | 0.77     |
| One-Hot-Encoded Categorical (Multiclass)  | 0.755    |

Following the initial performance analysis, we optimized the feature set. By selecting the optimal combination and number of features, we discovered:

| Optimal Number of Features | Feature Types Included       | Validation Accuracy |
|----------------------------|------------------------------|---------------------|
| 35                         | Mostly Numerical & Binary    | 0.89                |

### Model Selection
1. Our first model (for Milestone 4) following our logistic regression model is going to be a deep neural net (DNN). Our reasoning for this is because our target class is categorical with three separate categories to determine between. A neural net would allow us to create a model that outputs predictions for all three categories.
2. Our second model is going to be a support vector machine (SVM). Our plan for this is to combine the "Enrolled" and "Graduate" categories so that we can have a categorical class that can be represented in a binary fashion. This way, we can use the SVM and compare it with our original logistic regression model to compare and contrast the models.

### Milestone 3 Conclusion
Based on the analysis of our logistic regression model trained on the dataset, it appears to perform reasonably well, achieving satisfactory accuracy and loss metrics on both the training and validation data. The model effectively leverages carefully selected features to make accurate predictions regarding whether students drop out at the end of the semester or continue their enrollment and/or graduate.

However, while the model demonstrates promising performance, there are still opportunities for improvement. One such avenue is the exploration of regularization techniques. Regularization can be beneficial in mitigating the risk of overfitting, particularly in scenarios where there's uncertainty about the relevance of certain features or the presence of noise in the data. By introducing regularization, such as L1 (lasso) or L2 (ridge) regularization, we can encourage the model to learn simpler patterns that generalize better to unseen data, thus potentially enhancing its overall performance and robustness.

## Model 2: Deep Neural Network

### Evaluation of Data, Labels, and Loss Function

In terms of our data, we believe the preprocessing from the previous model is still ideal. We think it is appropriate to one-hot encode categorical variables because they should be treated as independent features (as opposed to one label-encoded feature). Additionally, we think some form of scaling is needed for the numerical data, so we decided to use the min-max scaled data which is been squeezed between the range from 0 to 1 to improve our model's performance. We also will continue to use the features which were produced the best result in our previous model because we predict these features have the highest correlation with a student's dropout rate. 

In terms of the loss function, our previous model, logistic regression, uses log loss. In this model, we decided to use binary cross-entropy loss since we are attempting a binary classification (dropout or not dropout). During hyperparameter tuning, we also experimented with using mean-squared error, but we found this loss function to be less effective than binary cross-entropy.

### Evaluation of Training vs. Testing Error

In our final neural network model evaluation, we have noted the following accuracies and loss values:

| Data Type    | Accuracy | Loss |
|--------------|----------|------|
| Training     | 0.88     | 0.31 |
| Validation   | 0.87     | 0.33 |
| Testing      | 0.86     | 0.35 |

Through K-fold cross-validation, we confirmed minimal discrepancy between training, validation, and testing errors, which implies that our model maintains its generality without overfitting.

### Model Comparison
When compared to our earlier logistic regression model, the neural network exhibits virtually identical performance in terms of training and testing accuracy. This suggests that both models are equally efficient at leveraging the selected features to predict student outcomes (dropouts or non-dropouts).

### Model Fitting

To evaluate our model's fitting we looked at the training and validation error over multiple epochs. Initially after hyperparameter tuning, we found that although we were able to achieve reasonable accuracies, it appeared as though our model was overfitting. This is because while the training error continuously decreased, the validation error decreased initially but then began to increase. This suggests that the model began to fit, but after a certain point, it became more accurate with respect to the training data but less generalizable to other datasets. This was also demonstrated by the testing data which had comparatively low accuracy to the training data. 

For our final model, we again evaluated the model fitting by plotting the training and validation error over multiple epochs. This time we found that the model seemed to accurately fit the data and was generalizable. Specifically, the change in loss for the training data closely matched the change in loss for the validation data. This indicates that the increase in accuracy during training was not specific to the training data, but rather indicative of generalizable learning. Again, this was also reflected in our testing data which achieved a similar accuracy to the training data.

### Hyperparameter Tuning, K-Fold Cross Validation, and Feature Expansion

#### Hyperparameter Tuning
We refined our neural network model's performance using `keras_tuner.RandomSearch`. The tuning focused on several key hyperparameters:
- Number of units in each layer: Ranged from 32 to 512, with a step size of 32
- Activation function: Included options were sigmoid, relu, tanh, and softmax
- Optimizer: SGD and RMSprop
- Learning rate: From 1e-4 to 1e-2

The optimal configuration was found to be:
- 224 units in each hidden layer
- tanh activation functions
- SGD optimizer
- Learning rate of approximately 0.002
- Loss function: Binary cross-entropy

This configuration yielded a training accuracy of 0.96, a validation accuracy of 0.83, and a testing accuracy of 0.84.

#### Identifying Overfitting
Despite high training accuracy, the significant gap with validation/testing accuracies suggested overfitting. To address this, we utilized:

#### K-Fold Cross Validation
We employed a 10-fold cross-validation method, leading to a more generalizable model. This best model demonstrated:
- Training accuracy: 0.88
- Validation accuracy: 0.87
- Testing accuracy: 0.86
- Loss: 0.31 (Training), 0.33 (Validation), 0.35 (Testing)

This approach helped us mitigate overfitting while maintaining strong model performance across all data sets.

### Model Selection
Our third and final model (for Milestone 5) is still going to be a support vector machine (SVM). We believe an SVM will do a good job at splitting the data into two classes (dropout and non-dropout) and we would like to experiment with different kernel functions to see if we can achieve a higher accuracy.

### Milestone 4 Conclusion
In the project, we developed two predictive models: a logistic regression model and a deep neural network, and the purpose is to predict student dropout rates. The logistic regression model, our first approach, provided a baseline with notable strengths in interpretability and speed. The second model, a deep neural network, represented a more advanced approach, leveraging a sequential architecture with multiple dense layers. This model showed a marked improvement, particularly in handling non-linear patterns, with validation and test accuracy scores improving to 87% and 86%, respectively. This suggests that the deep neural network could capture complex interactions between student features more effectively than logistic regression. To further enhance the neural network model, we could experiment with additional layers, different activation functions, or more sophisticated regularization techniques to combat overfitting, as indicated by the gap between training and validation performance. Incorporating dropout layers or exploring different optimization algorithms could also yield performance gains. Comparatively, the deep neural network performed better than the logistic regression model in terms of handling complex data structures, as evidenced by higher accuracy and more nuanced pattern recognition in the dropout predictions. However, the logistic regression model maintained its value through ease of use and interpretation, crucial for stakeholders to understand the contributing factors to student dropout.

## Model 3: Support Vector Machine

### Evaluation of Training vs Testing Error

After hyperparameter tuning, our best SVM achieved an accuracy of 0.86, a precision of 0.87, and a recall of 0.70. Our training data achieved a slightly higher accuracy of 0.9, as well as a higher precision and recall of 0.92 and 0.73 respectively. Since our metrics for our testing data is quite similar to our training data (just slightly under across all fronts), our SVM did not underfit to the data nor did it overfit. Our SVM performed quite well in predicting the labels of dropped-out and enrolled/graduate and generalized well to unseen data.

### Model Fitting

Similar to how we performed model fitting in our logistic regression model, we looked at the accuracy and loss for our training and validation data across a different number of features. As seen in the fitting graph, we concluded that our model improved dramatically in these aspects as we increased the number of features from 0-50. Our SVM's performance flatlined after that, however, and increasing the number of features used beyond 50 did not have a significant effect. This model resulted in the metrics listed above in our evaluation of the training and testing error.

### Hyperparameter Tuning

To optimize our model and maximize its performance in terms of accuracy and loss, we performed hyperparameter tuning using ```sklearn.model_selection.GridSearch```. The result of our hyperparameter tuning was C=100 (the highest regularization parameter we tested), kernel = polynomial, and degree = 1. This combination of parameters suggests that our SVM favored simplicity in the model over complexity. As our regularization parameter was the biggest we tested, complexity was punished in our model, and our SVM prevented overfitting to the training data and captured the overall pattern of the data well.

This is seen when we compared the training and testing errors. The model's performance on the testing data was very close to the training data, with only a slight dip in accuracy between the two. At an accuracy of 0.86, our model's performance held up well when it came to unseen data.

### Milestone 5 Conclusion
For our last model, we developed an SVM. Compared to the first model, our SVM performed extremely similarly and followed the same trends in terms of the number of features and the corresponding accuracies and losses to our logistic regression model. Our SVM performed similarly to our second model as well, which was our neural network. Our SVM was different compared to our neural network in the way that it was a lot simpler - with our optimal parameters punishing complexity in favor of more simple models (C = 100 as our regularization parameter, degree = 1 for our polynomial kernel).

Overall, our SVM performed comparatively well when it came to our accuracy and loss metrics for both our logistic regression model and our neural network. Our SVM is a simple classification model compared to our neural network, which utilizes different layers, activation functions, and other methods to predict our class target (dropped out vs enrolled/graduated).

## Results
Our project aimed to predict students' dropout and academic success based on a dataset from UC Irvine's Machine Learning Repository. After preprocessing the dataset, we applied three predictive models to predict our target variable, including **logistic regression**, a **deep neural network (DNN)**, and a **support vector machine (SVM)**.

### Model 1: Logistic Regression
The logistic regression model achieved an accuracy of 0.89, precision of 0.89, recall of 0.85, and loss of 3.52 on the training data. On the testing data, our model achieved an accuracy of 0.86, precision of 0.86, recall of 0.83, and loss of 5.07. The minimal difference between the training and testing data suggests an absence of overfitting for our model. 

#### Testing Data Metrics Report
|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| 0.0      | 0.86      | 0.94   | 0.90     | 569     |
| 1.0      | 0.87      | 0.72   | 0.79     | 316     |
| Accuracy |           |        | 0.86     | 885     |
| Macro avg| 0.86      | 0.83   | 0.84     | 885     |
| Weighted avg | 0.86  | 0.86   | 0.86     | 885     |

We also analyzed the impact of the number of features and the data type on model performance.

#### Model Fitting Graph (Fig. 1)
Impact of Number of Features on Model Performance
![loss_graph_logistic_regression](https://github.com/tcl002/CSE151AProject/blob/98ee683fd2b81cc18b313a54794dec4d9ef4246d/graphs/Logistic%20Regression/LossAcc%20-%20Num%20of%20Features.png)

As illustrated in the graphs above, both loss and accuracy plateaued beyond a certain threshold. The loss for both the training and validation datasets dramatically decreased as the number of features increased to approximately 50 features. Correspondingly, accuracy improved significantly for both the training and validation datasets as the number of features increased to approximately 50. Beyond that threshold, additional features would not contribute significantly to the model's performance.

#### Model Fitting Graph (Fig. 2)
Impact of Data Type on Model Performance
![features_graph_logistic_graph](https://github.com/tcl002/CSE151AProject/blob/98ee683fd2b81cc18b313a54794dec4d9ef4246d/graphs/Logistic%20Regression/LossAcc%20-%20Types%20of%20Features.png)

Additionally, the graphs above compare the model's performance across different data types, including numerical, categorical, binary, and one-hot encoded categorical (multi-class). It appears that the model performs best on numerical data, as it achieves the lowest loss and highest accuracy, followed then by binary, one-hot categorical, and categorical data, respectively.

### Model 2: Deep Neural Network (DNN)
The DNN model reported a training accuracy of 0.88, validation accuracy of 0.87, and testing accuracy of 0.85, with the loss being 0.31, 0.33, and 0.35, respectively. K-fold cross-validation helped ensure the model's generalizability by minimizing the difference between the training, validation, and testing errors.

#### Testing Metrics Report
|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| 0.0      | 0.84      | 0.95   | 0.89     | 569     |
| 1.0      | 0.88      | 0.67   | 0.76     | 316     |
| Accuracy |           |        | 0.85     | 885     |
| Macro avg| 0.86      | 0.81   | 0.83     | 885     |
| Weighted avg | 0.85  | 0.85   | 0.84     | 885     |

When constructing our DNN model, we experimented with both hyperparameter tuning and K-fold cross-validation.

#### Model Fitting Graph - (Hyperparameter Tuning)
![overfitting_neural_net](https://github.com/tcl002/CSE151AProject/blob/98ee683fd2b81cc18b313a54794dec4d9ef4246d/graphs/Neural%20Net/trainvaildloss.png)

In the graph above, there is a clear divergence between the training and validation loss after hyperparameter tuning which is indicative of overfitting. 

#### Model Fitting Graph - (Hyperparameter Tuning + K-Fold Cross Validation)   
![second_neural_net](https://github.com/tcl002/CSE151AProject/blob/98ee683fd2b81cc18b313a54794dec4d9ef4246d/graphs/Neural%20Net/avgloss.png)

To address the possibility of an overfitted model, we introduced K-fold cross-validation alongside hyperparameter tuning, in which we see a similar trend between the training and validation loss, which is indicative of a well-generalized model. 

### Model 3: Support Vector Machine (SVM)
The SVM model achieved an accuracy of 0.86, a precision of 0.85, and a recall of 0.83 for the testing data, and achieved an accuracy of 0.9, a precision of 0.90, and a recall of 0.86 for the training data. Optimal performance was achieved with C=100, a polynomial kernel, and a degree of 1.

#### Testing Metrics Report
|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| 0.0      | 0.86      | 0.93   | 0.89     | 569     |
| 1.0      | 0.85      | 0.72   | 0.78     | 316     |
| Accuracy |           |        | 0.86     | 885     |
| Macro avg| 0.85      | 0.83   | 0.84     | 885     |
| Weighted avg | 0.85   | 0.86   | 0.85    | 885     |

#### Model Fitting Graph
![svm_graph](https://github.com/tcl002/CSE151AProject/blob/1ebaa31fb0235fecea731e2267baae88c268d211/graphs/svm/lossandacc.png)

Overall, all three models performed comparably well in predicting students' dropout rates and academic success, with each model having distinct strengths that allowed us to comprehensively analyze the dataset. 

## Discussion
We began this project with a meticulous exploration of our dataset to understand its intricacies. We recognized the significance of data preprocessing as the foundation of our model's performance. Encoding categorical variables, normalizing numerical variables, handling missing data, and conducting correlation analysis were crucial steps to ensure data quality and enhance the model's predictive power.

_Data Preprocessing:_ We utilized OneHotEncoding for categorical variables and MinMaxScaler for numerical variables. These techniques aimed to standardize the data and make it suitable for modeling. However, despite our efforts, we acknowledge that preprocessing is an iterative process, and there may be alternative methods or additional considerations we could explore in future iterations.

_Model Development:_ Our choice of models—logistic regression, deep neural network (DNN), and support vector machine (SVM)—was driven by the nature of our target variable and the complexity of the data. Logistic regression provided a baseline, offering interpretability and speed. However, its simplicity limited its ability to capture complex patterns in the data. The DNN, on the other hand, demonstrated superior performance in handling non-linear relationships but required more computational resources and hyperparameter tuning. The SVM presented a balance between simplicity and performance, although its interpretability was not as straightforward as logistic regression.

Throughout our development, we faced challenges such as determining the balance between model complexity and performance. For example, our DNN's initial overfitting prompted the discussion of the importance of validation techniques, whereas our logistic regression and SVM models highlighted the value of simplicity and feature selection.

_Evaluation and Interpretation:_ Throughout our analysis, we continuously evaluated our models' performance on both training and testing data to ensure generalizability. We observed minimal differences between training and testing errors for all models, indicating that overfitting was effectively mitigated. This instilled confidence in the reliability of our results and the generalizability of our models.

_Believability of Results:_ While our models achieved promising results, each of them has its assumptions and constraints, which may not fully capture the complexity of real-world scenarios. Additionally, our analysis is based on a specific dataset, and the generalizability of our findings to other contexts should be approached with caution. Furthermore, the dynamic nature of data and the possibility of unforeseen variables emphasize the importance of ongoing refinement and validation of our models, as student populations vary and as the educational landscape continues to evolve.

## Conclusion
Our analysis revealed the importance of data preprocessing in enhancing model performance. Each step, such as encoding categorical variables and handling missing data,  played a critical role in ensuring the quality and integrity of our predictions. Furthermore, our exploration of different modeling techniques—logistic regression, deep neural networks, and support vector machines—provided valuable insights into the complexity of the problem and the trade-offs between interpretability and performance.

Moving forward, there are several avenues that we can explore and refine. One area of focus is the incorporation of advanced regularization techniques, such as dropout layers in neural networks or kernel methods in support vector machines, to combat overfitting and improve model generalizability. Additionally, the exploration of ensemble methods, such as random forests or gradient boosting, could leverage the strengths of multiple models to enhance predictive accuracy and robustness.

Finally, as we conclude this project, it's important to recognize that predictive modeling is an iterative process, where each iteration builds upon the lessons learned from previous attempts. While our models provide valuable insights, they represent just one snapshot in time, and there's always room for refinement and enhancement. What we strive for is to gain a more accurate and robust predictive model so that we can provide information in educational settings that can prompt actionable intervention.

## Collaboration

##### Kim
During my involvement in the project, I made significant contributions to both the code implementation and the write-up. My contributions can be summarized as follows:

Developed parts of the data preprocessing tasks, such as the displaying of the correlation heatmap and selection of features.
Implemented parts of the machine learning models, including logistic regression, deep neural networks, and support vector machines. My contribution is heavily in the DNN however.

Authored sections of the write-up covering a part of the introduction, milestone 3 conclusion, discussion, and most of the conclusion as well as provided the testing metrics report in the result section.

I attended our meetings as well to discuss our next goals or milestones. Concurrently, I've provided some critical analysis and interpretation of our model performance metrics, relayed my thoughts to some of the strengths, and weaknesses or our models, and wrote what opportunities for improvement we can endeavor.

##### Tyler
I attended each weekly meeting to communicate with the team about what was still required and how to separate tasks for the upcoming milestone. I contributed to parts of the writeup, including but not limited to the following sections: abstract, model selection, model performance analysis, etc. I also contributed to the code, including but not limited to the following sections: data visualization, data preprocessing, neural net, support vector machine, etc.

##### Andrew
I set up the group logistics, including meetings, group discord, and setting up the GitHub. I helped with the development of parts of the models, primarily the Logistic Regression model. For the writeup, I helped discuss what was needed for each section as well as how to go about filling in each section, specifically how to go about evaluating each model. Alongside setting up the meetings, I also attended each one providing ideas on what to add or change to our existing notebook at each task.

##### Kate
I attended the weekly group meetings to discuss our progress on the projects and plan next steps. For milestone 1, I made helped come up with ideas for our project in our first group meeting and also made revisions to the abstract. For milestone 2, I did a large portion of the data processing including splitting the data into numerica / categorical / binary, one hot encoding the target, and min-max scaling numerical data. I made the heatmap of correlations (see above figure under data exploration) as well as the stripplot to visualize numerical variables and the barplots to visualize categorical / binary variables. For milestone 3, I implemented model fitting by first running our logistical regression on binary / numerical / categorical / one-hot-categorical variables and then fitting the number of features (and graphing the loss). I also contributed to the writeup. For the 4th and 5th milestones I also contributed to the writeup. For instance, for the 4th milestone I wrote the section for evaluation of data, labels, and loss function as well as model fitting. 

##### James
I attended the weekly group meetings where we went over our planning for our project, and contributed to brainstorming how we wanted to approach our project, both in terms of actual project content and group organization (meetings). I helped preprocess the data and also implemented the logistic regression model that we used in Milestone 3 and tested it on our training and validation and testing data. I also wrote up the analysis for our third model, our SVM. Additionally, I contributed to the writeup, namely the methods and results section.

##### Emily 
I attended weekly group meetings to discuss the progress of our project, and what our next steps + action items were for the upcoming week. I helped select our dataset based on its features and its relevance to us as students. Project-wise, my contributions primarily lie in the writeup, where I worked namely on the methods and results section, analyzing our findings to conclude our project. I also helped discuss the role of hyperparameter tuning for our second model and assisted with the fine-tuning of our final milestone, including downloading images from our notebook to include in our final write up.  

### Ka Wing Yan:
In the project beginning, I helped to write the abstract and discuss which dataset to choose And attend weekly meeting. Next, Start Coding about the data exploration and data processing (loading data, diagram building, missing data checking, listing the top 10 in relation to the target, and plotting out the graph). Build up early vision LogisticRegression and RandomForestClassifier for testing dataset. Set up a local environment write a script in requirements.txt and let the code run in a local machine and how to install conda.  Discuss how to build up the logistic regression, deep neural network, and SVM(Support Vector Machine).Write a milestone 4 conclusion comparing the differences between the two models And discuss what is the next model.
