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

**BalancedRandomForestClassifier**
* Balanced accuracy score: 78%
* Precision: 4% (high risk), 100% (low risk)
* Recall: 67% (high risk), 91% (low risk)

Confusion matrix:

<img width="358" alt="Screen Shot 2021-09-19 at 12 14 38 PM" src="https://user-images.githubusercontent.com/45336910/133934781-39232aa8-7412-464c-97cb-779d21670ca6.png">

Classification report:

<img width="659" alt="Screen Shot 2021-09-19 at 12 14 54 PM" src="https://user-images.githubusercontent.com/45336910/133934794-618e1d55-34b8-4e4b-87f3-72dafadc256b.png">

Top 5 important features:
1. total_rec_prncp
2. total_rec_int
3. total_pymnt_inv
4. total_pymnt
5. last_pymnt_amnt

**EasyEnsembleClassifier**
* Balanced accuracy score: 92%
* Precision: 7% (high risk), 100% (low risk)
* Recall: 91% (high risk), 94% (low risk)

Confusion matrix:

<img width="365" alt="Screen Shot 2021-09-19 at 12 16 41 PM" src="https://user-images.githubusercontent.com/45336910/133934860-2ba5ca34-5c7f-4588-a1fa-47141bd26e83.png">


Classification report:

<img width="656" alt="Screen Shot 2021-09-19 at 12 16 54 PM" src="https://user-images.githubusercontent.com/45336910/133934866-2fde13c0-fd8c-47c4-a29f-9552a81c84b8.png">

## Summary and Analysis
* The ensemble classifiers have higher balanced accuracy scores and show increased precision for high risk candidates when compared to the resampling methods.
* Ensemble classifiers have overall higher f1 scores (which considers both the influence of precision and recall)
* For this application, banks would most likely not want to incorrectly classify high/low risk candidates and therefore focusing on only recall or only precision is unhelpful. Therefore, I would not recommend any of these models to the bank for credit risk predicition because they all seem to do a poor job of correctly identifying high risk candidates. 
* 
