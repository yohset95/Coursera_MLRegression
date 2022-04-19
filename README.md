# Flight Ticket Price Prediction using Linear Regression Models
This is my final project of Coursera's IBM Machine Learning: Regression. </br>
Written by Yohanes Setiawan.

# **Business Understanding**

## Introduction
- 'Easemytrip' is an internet platform for booking flight tickets, and hence a platform that potential passengers use to buy tickets 
- A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers

## Problem Statement

Passengers are difficult to calculate a range of ticket price to make better plan for their trip

## Goal

To give feedbacks to passengers in India for their best trip planning and predict the ticket price based on given features in Easemytrip application

## Research Questions
- How is the price affected when tickets are bought in just 1 or 2 days before departure?
- Does ticket price change based on the departure time and arrival time?
- Does price vary with Airlines?
- How does the ticket price vary between Economy and Business class?
- How to predict flight ticket price?

## Objective Statements

- Get insight about the effect of ticket price which are bought in just 1 or 2 days before departure
- Get insight about ticket price change based on the departure time and arrival time
- Get insight about variations of ticket price with airlines
- Get insight about variation of ticket price between economy and business class
- Conduct research to find the best model of flight price prediction using Linear Regression Models for passengers in India

# **Analytical Approach**

- Descriptive analysis
- Graph analysis
- Table analysis
- Predictive analysis (Regression Problem)

# **Data Understanding**

- Data source was secondary data and was collected from Ease my trip website
- Source: https://www.kaggle.com/shubhambathwal/flight-price-prediction
- A total of 300261 distinct flight booking options was extracted from the site
- Data was collected for 50 days, from February 11th to March 31st, 2022

## Dataset Information
![](https://drive.google.com/uc?export=view&id=1LC33M9acmR9y14oTHOnwFNVVoPMMIZ9k) </br>
This means there is no missing value in the dataset

## Checking Duplicated Data
![](https://drive.google.com/uc?export=view&id=17ZLr2t0N35BnxLfbIm0W0EqA34GmznOI) </br>
There is no duplicated data in the dataset.

# Exploratory Data Analysis (EDA)

## Descriptive Statistics
For numerical columns/features: </br>
![](https://drive.google.com/uc?export=view&id=17HNCtDKWpbBnBODAezFelaADbwXzZDxK) </br> </br>
For categorical columns/features: </br>
![](https://drive.google.com/uc?export=view&id=17HNCtDKWpbBnBODAezFelaADbwXzZDxK) </br>
- The most favorite airline is Vistara
- The passengers mostly depart from Delhi 
- The most likable destination city is Mumbai
- The passengers are likely to choose one stop for time efficiency
- Morning departure time has been the best time for passengers
- The passengers like to arrive at night
- There are 1561 unique values in column "flight". Therefore, it should be removed to avoid redundancy feature


## Univariate Distribution of Observations
![](https://drive.google.com/uc?export=view&id=13EENW6HLJhmITmSmZHUdTrCVWo5ZwpyL) </br>
- Uniform distribution: "Unnamed: 0" and "days_left"
- Skewed: "duration", "price"

## Box Plot
![](https://drive.google.com/uc?export=view&id=1Hltq3DlHLRw7LIgLLXCEeuw6C2PLE40S) </br>
- The only independent features/columns with outlier is "duration"
- "price" is the dependent feature with outlier. However, this can be ignored because "price" is our target variable
- Passengers with duration between cities for more than 30 hours are considered as outliers

## Categorical Plot
![](https://drive.google.com/uc?export=view&id=1A2rT5CCtxEc0HPg5THnSzACnVrnFrY74) </br>

## Correlation
![](https://drive.google.com/uc?export=view&id=1Y59UitQVnl0QjcKzOoPh1iCbjVFBlbGe) </br>
* Column "Price" is highly correlated with "Unnamed: 0". This means the higher values in "Unnamed: 0", also the higher values in "Price" as the target variable
* However, the "Unnamed: 0" is the ID of the customer, which is sorted from the lowest bought ticket price until the highest bought ticket price
* Therefore, column "Unnamed: 0" will affect the predicted model seriously and need to be removed

## D'Agustino K^2 Normality Test For Target Variable
- alpha = 0.05 (5%)
- H0 (Null Hypothesis): Column "Price" is normally distributed
- H1 (Alternative Hypothesis): Column "Price" is not normally distributed
- The null hypothesis will be rejected if p-value < alpha 

![](https://drive.google.com/uc?export=view&id=1fwDfLLnJPJt7G4t9wuWSp9xGTAHNkU7X) </br>

* Because the p-value equals to 0, which rejects the null hypothesis (p-value=0<0.05=alpha), our target variable (Price) is not normally distributed
* Finally, the target variable will be transfromed using Box Cox Transformation
* However, the result from non Box Cox Transformation is still conducted in order to compare between the Box Cox Transformation
* The model with the best result will be chosen for the next stage of further regression models

## Get Insights
### How is the price affected when tickets are bought in just 1 or 2 days before departure?

![](https://drive.google.com/uc?export=view&id=1MoHZmtp4JBMwkqTO45cXjv-dRu1z4blF) </br>

### Does ticket price change based on the departure time and arrival time?

![](https://drive.google.com/uc?export=view&id=1FNMyVdIm7CBRspygu-bqjfvOuWspVxIq) </br>

### How the price changes with change in Source and Destination?

![](https://drive.google.com/uc?export=view&id=1BgzOjnJ9xzz1PfRf1NUnKu14gdamscHz) </br>

### Does price vary with Airlines?

![](https://drive.google.com/uc?export=view&id=1ARHwNU4X1FJJ0fFPOOF0IuEaZQH_aNnH) </br>

* The ticket prices vary with airlines 
* Air India and Vistara are the most varying airline because of its highest Coefficient of Variation (CV)

### How does the ticket price vary between Economy and Business class?

![](https://drive.google.com/uc?export=view&id=1hT3jYbjg96UKj0GFLJp9YGE6J71WCALA) </br>

![](https://drive.google.com/uc?export=view&id=17-uoxfYVdrT-DmS3THB_LsWpFLIJVEjU) </br>

### Does ticket price change based on the number of stops between the source and destination cities?

![](https://drive.google.com/uc?export=view&id=1M86YcZiELlm-xD783jzqXMbfAa89yNqk) </br>

# **Feature Engineering**
* Handling Outlier using Interquartile Range Analysis: `duration`
* Removing Irrelevant Feature(s): `Unnamed: 0` and `flight`
* Categorical Encoding using One Hot Encoding for categorical columns

# **Data Preparation**
* Training-Testing Split: 70% training data and 30% testing data
* Feature Scaling with StandardScaler()
* Box Cox Transformation for Target Variable

# **Modelling**
* I used 3 linear regression models: Multiple Linear Regression, Polynomial Regression, and Lasso Regression
* The model will be evaluated by 3-Fold Validation to check whether the model is overfitting or underfitting

## Multiple Linear Regression
* Generalization of simple linear regression for more than one predictor variable. </br>
* Two multiple linear regression are compared: With and Without Box Cox Transformation
* With Box Cox Transformation: </br>
![](https://drive.google.com/uc?export=view&id=1crar0irO-_2ThLJAPq1WDKrxSluoGVF2) </br>
* Therefore, the model from Multiple Linear Regression with Box Cox Transformation is not overfitting or underfitting
* Without Box Cox Transformation: </br>
![](https://drive.google.com/uc?export=view&id=1l-AQxmRPAZ7Z7YkcEJOjtkKVzN6nuQDO) </br>
* Therefore, the model from Multiple Linear Regression without Box Cox Transformation is not overfitting or underfitting
* Because the error from Multiple Linear Regression without Box Cox Transformation is lower than with Box Cox Transformation, Multiple Linear Regression without Box Cox Transformation will be chosen to compare with other linear regression models

## Polynomial Regression
* Linear Regression with Polynomial Features
* I choose the maximum degree of polynomial = 2 because the dataset have too many features such that the polynomial transformation of the features will not affect too many addition to the features of the dataset. </br>
![](https://drive.google.com/uc?export=view&id=1EH2hpMNK48O9XrPdM0X5jZnSwUdJsqBi) </br>
* Therefore, the model from Polynomial Regression is not overfitting or underfitting
* Because the model with polynomial features are better than without polynomial feature, then the writer will add polynomial features in Lasso Regression

## Lasso Regression
* Linear regression which performs shrinkage regularization with automatically selecting features
* Lasso has one important hyperparameter, that is `alpha`. To find the best `alpha`, I used hyperparameter tuning (`GridSearchCV`) for training the model. </br>
![](https://drive.google.com/uc?export=view&id=1Aqy5vQIms9uRR_FR4z662d4WrGEj_p_J) </br>
* It can be seen that degree of polynomial features = 2 has the higher R squared Score
* The best hyperparameter: `polynomial_features___degree=2` and `alpha=0.20158909717702692`
* Model with the chosen `alpha` will be evaluated by 3-Fold Validation as previous models:
![](https://drive.google.com/uc?export=view&id=1IN5l3Bu66zLvn2NMF0YFHdJd-LxMEBFM) </br>
* Therefore, the model from Lasso Regression is not overfitting or underfitting

# Model Evaluation
* I plot the scatter plot for every model and evaluate every model using R Squared (R2) Score
* In addition, Root Mean Square Error (RMSE) is used to measure error between predicted and actual values

## Multiple Linear Regression
![](https://drive.google.com/uc?export=view&id=1k3EfBkyqmWfHzO9FUz1bddAWoIBj6GWz) </br>
As we have seen that Multiple Linear Regression has failed to capture flight ticket price in range [0,50000) which is lower price

## Polynomial Regression
![](https://drive.google.com/uc?export=view&id=1lbp9UpesLi8mC6G6Euoqg_i0xTh59xRl) </br>
The model from Polynomial Regression is good enough to predict ticket prices

## Lasso Regression
![](https://drive.google.com/uc?export=view&id=16n38c1fCsd-dO-pRBJ1eqYo44kjWb4Hb) </br>
Lasso regression model tends to similar with Polynomial Regression. It can captures model very well. However, in order to compare between Polynomial and Lasso, I need to check the evaluation between models.

## Model Selection
Type | RMSE | R2 Score |
--- | --- | --- | 
Multiple Linear Regression | 6774.180186 | 0.911250
Polynomial Regression | 5011.836200	| 0.951421
Lasso Regression | 5010.349536 | 0.951450

* In summary, Lasso Regression has been chosen to be the best linear regression 
model to predict flight ticket price because it has the lowest RMSE and highest R2 Score
* For future predicitons, Lasso will be saved in a pickle form which is ready to be deployed

# Get Insights from the Selected Model
* Length of features (with Polynomial Features, not Lasso) is 741 features
* Length of features after Feature Selection in Lasso is 197 features
* I have many redundant features through polynomial features which is automatically removed by Lasso for better prediction results
* I also plot our Top 10 Feature Importances for Flight Ticket Price Prediction: </br>
![](https://drive.google.com/uc?export=view&id=1eyT8VhyViYrd-Meq183bojp_E5ubrIp5) </br>
* Business Class has been the most important feature in Lasso's prediction because it is a special class which exists only in Air India and Vistara. The chosen class determines the ticket prices in prediction
* Duration is important to determine the ticket price. Therefore, customer should consider the overall amount of time it takes to travel between cities before ordering flight tickets

# Summary of Findings and Suggestions

* If passengers want cheaper tickets, they should buy around 25-30 days before departure. Buying tickets 1-2 days before departure is only for emergency
* The best departure for cheaper tickets occures when passengers choose Late Night or Early Morning as departure and arrival time
* Bangalore, Chennai, and Kolkata are the top 3 highest ticket price
* Delhi and Hyderabad are considered as top 2 lowest ticket price
* If passengers want to try business class in a cheaper mode, then they should choose Air India. But, if passengers choose the most likable business class with best facilities (and higher ticket price absolutely), they may choose Vistara
* If passengers want cheaper ticket, they should choose flight higher duration between cities. The more less duration between cities, the higher ticket price should be
* The standard linear regression model is severely under performing on low and high valued tickets, while the polynomial and ridge models are smoothly fit across the entire range of ticket prices.
* Lasso regression with alpha between 0 and 1 has been the best alpha for modelling by searching from hyperparameter tuning (GridSearch)
* Lasso regression tends to remove more features in using the polynomial features
* Showing the list of features the model believes are the most important in predicting the ticket price to give insights


For coding implementation, the `.ipynb` file is available to read and download in this repository. </br>
For further details of presentation, the `.pdf` file is given with method explanations and dataset insights (without coding) </br>
Thank you for reading! </br>
Have a nice day.