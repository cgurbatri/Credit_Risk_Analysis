# Credit Risk Analysis

## Project Description
The purpose of this analysis is to use supervised machine learning models to predict credit risk to provide quicker and more reliable loan experiences. Ideally the machine learning algorithm will aid in predicting good candidates for loans, which can ultimately lead to lower default rates. 

**Resources** 
Data: LoanStats_2019Q1.csv
Software: Python 3.8.5, Jupyter Notebook 6.1.4, conda 4.10.3

**Method**

Preprocessing: 
* Drop columns and rows where all values are null 
* Remove "Issued" loan status
* Convert interest rate column to numerical
* Target column (loan_status) conversion to low_risk and high_risk
* Encode categorical variables using Pandas and ScikitLearn

Resampling to address class imbalance:
* RandomOverSampler and SMOTE (oversample the data)
* ClusterCentroids (undersample the data)
* SMOTEENN (combination of under/oversampling of the data)

Classifiers:
* BalancedRandomForestClassifier
* EasyEnsembleClassifier

Evaluation metrics:
* Confusion matrix
* Balanced accuracy score
* Imabalanced classification reports

## Results

**RandomOverSampler**
* Balanced accuracy score: 65%
* Precision: 1% (high risk), 100% (low risk)
* Recall: 62% (high risk), 68% (low risk)

Confusion matrix: 

<img width="363" alt="Screen Shot 2021-09-19 at 12 03 38 PM" src="https://user-images.githubusercontent.com/45336910/133934426-1121875c-9cb2-461b-a587-1591e47d0435.png">

Classification report: 
<img width="657" alt="Screen Shot 2021-09-19 at 12 02 52 PM" src="https://user-images.githubusercontent.com/45336910/133934393-25054f44-0d51-472f-bb25-3eeb482a9e40.png">

**SMOTE**
* Balanced accuracy score: 65%
* Precision: 1% (high risk), 100% (low risk)
* Recall: 64% (high risk), 66% (low risk)

Confusion matrix: 

<img width="361" alt="Screen Shot 2021-09-19 at 12 07 27 PM" src="https://user-images.githubusercontent.com/45336910/133934541-55ec5752-df38-418e-b3fd-410af4dc2beb.png">

Classification report:
<img width="653" alt="Screen Shot 2021-09-19 at 12 08 09 PM" src="https://user-images.githubusercontent.com/45336910/133934572-e22b76e6-bce3-47f8-8eb5-f9ae9fb150b6.png">

**ClusterCentroids**
* Balanced accuracy score: 51%
* Precision: 1% (high risk), 100% (low risk)
* Recall: 64% (high risk), 39% (low risk)

Confusion matrix:

<img width="356" alt="Screen Shot 2021-09-19 at 12 09 34 PM" src="https://user-images.githubusercontent.com/45336910/133934616-ef93a00b-2db6-4b26-b871-78d401dc7c4e.png">

Classification report:
<img width="650" alt="Screen Shot 2021-09-19 at 12 09 43 PM" src="https://user-images.githubusercontent.com/45336910/133934624-1e983220-42c0-45f4-ba44-bdfb4a05ba5e.png">

**SMOTEEN**
* Balanced accuracy score: 61%
* Precision: 1% (high risk), 100% (low risk)
* Recall: 69% (high risk), 55% (low risk)

Confusion matrix:

<img width="359" alt="Screen Shot 2021-09-19 at 12 10 36 PM" src="https://user-images.githubusercontent.com/45336910/133934650-909d2b80-97fe-4a60-aa37-86993e80acdc.png">


Classification report:
<img width="667" alt="Screen Shot 2021-09-19 at 12 10 47 PM" src="https://user-images.githubusercontent.com/45336910/133934657-5c8ba161-9e9e-4677-a2cc-45e0cfb97ddf.png">




