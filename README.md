<h1 align="center"> Creating a customer ranking system for an insurance company</h1>

<p align="center">A learning to rank cross-sell project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198294567-6a53415a-7b3e-48ea-ab58-482d849c6309.svg" alt="drawing" width="400"/>
</p>

*Obs: Business problem, company and data are fictitious.*

*The in-depth Python code explanation is available in [this](https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare.ipynb) Jupyter Notebook.*

# 1. **Insuricare and Business Problem**
<p align="justify"> Insuricare is an insurance company that has provided health insurance to its customers, and now they are willing to sell a new vehicle insurance to their clients. To achieve that, Insuricare conducted a research with around 305 thousand customers that bought the health insurance last year, asking each one if they would be interested in buying the new insurance. This data was stored in the company's database, alongside other customers' features. 

Then, Insuricare Sales Team selected around 76 thousand new customers, which are people that didn't respond to the research, to offer the new vehicle insurance. However, due to a limit call <i>restriction*</i> Insuricare must choose a way of selecting which clients to call: </p>

- Either select the customers randomly, which is the <b>baseline model</b> previously used by the company.
  
- Or, the Data Science Team will provide, by using a Machine Learning (ML) model, an ordered list of these new customers, based on their propensity score of buying the new insurance.

<i> * Insuricare Sales Team would like to make 20,000 calls, but it can be pushed to 40,000 calls. </i>

# 2. **Data Overview**
The training data was collected from a PostgreSQL Database. The initial features descriptions are available below:

| **Feature**          | **Definition** |
|----------------------|----------------|
| id                   | Unique ID for the customer|
| gender               |Gender of the customer|
| age                  |Age of the customer|
| region_code          |   Unique code for the region of the customer|
| policy_sales_channel |Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc|
| driving_license      |0 : Customer does not have DL, 1 : Customer already has DL|
| vehicle_age          |Age of the Vehicle|
| vehicle_damage       |Yes : Customer got his/her vehicle damaged in the past. No : Customer didn't get his/her vehicle damaged in the past.|
| previously_insured   |1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance|
| annual_premium       |The amount customer needs to pay as premium in the year|
| vintage              |Number of Days, Customer has been associated with the company|
| response             | 1 : Customer is interested in the new insurance, 0 : Customer is not interested in the new insurance|

# 3. **Business Assumptions and Definitions**

- Cross-selling is a strategy used to sell products associated with another product already owned by the customer. In this project, health insurance and vehicle insurance are the products. 
- Learning to rank is a machine learning application. In this project, we are ranking customers in a list, from the most likely customer to buy the new insurance to the least likely one. This list will be provided by the ML model.

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To provide an ordered list of these new customers, based on their propensity score of buying the new insurance the following steps were performed: </p>

- <b> Understanding the Business Problem </b> : Understanding the main objective Insuricare was trying to achieve and plan the solution to it. 

- <b> Collecting Data </b>: Collecting data from a PostgreSQL Database, as well as from Kaggle.

- <b> Data Cleaning </b>: Checking data types and Nan's. Other tasks such as: renaming columns, dealing with outliers, changing data types weren't necessary at this point. 

- <b> Feature Engineering </b>: Editing original features, so that those could be used in the ML model. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA) </b>: Exploring the data in order to obtain business experience, look for useful business insights and find important features for the ML model. The top business insights found are available at <a href="https://github.com/brunodifranco/project-insuricare-ranking#5-top-business-insights"> Section 5</a>. </p>

- <b> Data Preparation </b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/"> Rescaling Techniques</a> in the data, as well as <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Enconding Methods</a>, to deal with categorical variables. 

- <b> Feature Selection </b>: Selecting the best features to use in the ML model by using <a href="https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f"> Random Forest</a>. 

- <p align="justify"> <b> Machine Learning Modeling </b>: Training Classification Algorithms with cross-validation. The best model was selected to be improved via Bayesian Optimization with Optuna. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#6-machine-learning-models">Section 6</a>. </p>

- <b> Model Evaluation </b>: Evaluating the model using two metrics: Precision at K and Recall at K, as well as two curves: Cumulative Gains and Lift Curves. 

- <b> Results </b>: Translating the ML model to financial and business performance.

- <p align="justify"> <b> Propensity Score List and Model Deployment </b>: Providing a full list of the 76 thousand customers sorted by propensity score, as well as a Google Sheets that returns propensity score and ranks customers (used for future customers). This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#7-business-and-financial-results"> Section 7</a>. </p>
  
## 4.2. Tools and techniques used:
- [Python 3.10.8](https://www.python.org/downloads/release/python-3108/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Sklearn](https://scikit-learn.org/stable/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [Python API's](https://realpython.com/api-integration-in-python/).  
- [Render Cloud](https://render.com/), [Google Sheets](https://www.google.com/sheets/about/) and [JavaScript](https://www.javascript.com/).
- [Git](https://git-scm.com/) and [Github](https://github.com/).
- [Exploratory Data Analysis (EDA)](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15). 
- [Techniques for Feature Selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/).
- [Classification Algorithms (KNN Classifier, Logistic Regression; Random Forest, AdaBoost, CatBoost, XGBoost and LGBM Classifiers)](https://scikit-learn.org/stable/modules/ensemble.html).
- [Cross-Validation Methods](https://scikit-learn.org/stable/modules/cross_validation.html), [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) and [Learning to Rank Performance Metrics (Precision at K, Recall at K, Cumulative Gains Curve and Lift Curve)](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54).

# 5. **Top Business Insights**

 - ### 1st - Older customers are more likely to buy the new vehicle insurance.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198151697-f82a4e61-cbed-4465-849c-2cf81fd4762c.png" alt="drawing" width="800"/>
</p>

--- 
- ### 2nd - Customers with older vehicles are more likely to buy vehicle insurance.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198151788-1018458c-8e67-4ead-9d15-76622b4df287.png" alt="drawing" width="800"/>
</p>

--- 

- ### 3rd - Men are more likely to buy the new vehicle insurance than women.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198151862-4de5cab2-1647-4aae-bc9a-b0772e91ef18.png" alt="drawing" width="800"/>
</p>

---

# 6. **Machine Learning Models**

<p align="justify"> This was the most fundamental part of this project, since it's in ML modeling where we can provide an ordered list of these new customers, based on their propensity score of buying the new insurance. Seven models were trained using cross-validation: </p>

- KNN Classifier
- Logistic Regression
- Random Forest Classifier
- AdaBoost Classifier
- CatBoost Classifier
- XGBoost Classifier 
- Light GBM Classifier

The initial performance for all seven algorithms are displayed below:

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|    LGBM Classifier       | 0.2789 +/- 0.0003  |0.9329 +/- 0.001 |
|    AdaBoost Classifier   | 0.2783 +/- 0.0007	|0.9309 +/- 0.0023|
|      CatBoost Classifier | 0.2783 +/- 0.0005	|0.9311 +/- 0.0018| 
|   XGBoost Classifier     | 0.2771 +/- 0.0006  |0.9270 +/- 0.0022|
|    Logistic Regression   | 0.2748 +/- 0.0009  |0.9193 +/- 0.0031|
| Random Forest Classifier | 0.2719 +/- 0.0005  |0.9096 +/- 0.0016|
|      KNN Classifier      | 0.2392 +/- 0.0006  |0.8001 +/- 0.0019|
</div>

<i>K is either equal to 20,000 or 40,000, given our business problem. </i>

<p align="justify"> The <b>Light GBM Classifier</b> model will be chosen for hyperparameter tuning, since it's by far the fastest algorithm to train and tune, whilst being the one with best results without any tuning. </p>

LGBM speed in comparison to other ensemble algorithms trained in this dataset:
- 4.7 times faster than CatBoost 
- 7.1 times faster than XGBoost
- 30.6 times faster than AdaBoost
- 63.2 times faster than Random Forest

<p align="justify"> At first glance the models performances don't look so great, and that's due to the short amount of variables, on which many are categorical or binary, or simply those don't contain much information. 

However, <b>for this business problem</b> this isn't a major concern, since the goal here isn't finding the best possible prediction on whether a customer will buy the new insurance or not, but to <b>create a score that ranks clients in a ordered list, so that the sales team can contact them in order to sell the new vehicle insurance</b>.</p>

After tuning LGBM's hyperparameters using [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) the model performance has improved on the Precision at K, and decreased on Recall at K, which was expected: 

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|      LGBM Classifier     | 0.2793 +/- 0.0005  |0.9344 +/- 0.0017| 

</div>

## <i>Metrics Definition and Interpretation</i>

<p align="justify"> <i> As we're ranking customers in a list, there's no need to look into the more traditional classification metrics, such as accuracy, precision, recall, f1-score, aoc-roc curve, confusion matrix, etc.

Instead, **ranking metrics** will be used:

- **Precision at K** : Shows the fraction of correct predictions made until K out of all predictions. 
  
- **Recall at K** : Shows the fraction of correct predictions made until K out of all true examples. 

In addition, two curves can be plotted: 

- <b>Cumulative Gains Curve</b>, indicating the percentage of customers, ordered by probability score, containing a percentage of all customers interested in the new insurance. 

- <b>Lift Curve</b>, which indicates how many times the ML model is better than the baseline model (original model used by Insuricare). </i> </p>

# 7. **Business and Financial Results**

## 7.1. Business Results

**1) By making 20,000 calls how many interested customers can Insuricare reach with the new model?**
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152035-48c27ead-53f8-440e-af92-f049456dac33.png" alt="drawing" width="1000"/>
</p>

<p align="justify"> 

- 20,000 calls represents 26.24% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 71.29% of customers interested in the new vehicle insurance, since 0.7129 is our recall at 20,000. </p>

- As seen from the Lift Curve, our **LGBM model is 2.72 times better than the baseline model at 20,000 calls.** 

**2) Now increasing the amount of calls to 40,000 how many interested customers can Insuricare reach with the new model?**

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152040-929e3f17-d07e-401a-892c-50bf9c01f475.png" alt="drawing" width="1000"/>
</p>

- 40,000 calls represents 52.48% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 99.48% of customers interested in the new vehicle insurance, since 0.9948 is our recall at 40,000.

- At 40,000 calls, our **LGBM model is around 1.89 times better than the baseline model.**  

## 7.2. Expected Financial Results

To explore the expected financial results of our model, let's consider a few assumptions:

- The customer database that will be reached out is composed of 76,222 clients.
- We expect 12.28% of these customers to be interested in the new vehicle insurance, since it's the percentage of interest people that participated in the Insuricare research. 
- The annual premium for each of these new vehicle insurance customers will be US$ 2,630 yearly. *

*<i> The annual premium of US$ 2,630 is set for realistic purposes, since it's the lowest and most common value in the dataset. </i>

The expected financial results and comparisons are shown below:

<div align="center">

|    **Model**    |  **Annual Revenue - 20,000 calls** | **Annual Revenue - 40,000 calls** |  **Interested clients reached out - 20,000 calls** | **Interested clients reached out - 40,000 calls** |
|:---------------:|:---:|:-----------------------------------:|:---:|:---------------------------------------:|
|       LGBM      | US$ 17,515,800.00    |US$ 24,440,590.00          | 6660   |9293                  |
|     Baseline    |  US$ 6,446,130.00    |US$ 12,894,890.00           | 2451  |4903                  |
| $\Delta$ (LGBM, Baseline) |  11,069,670.00     |US$ 11,545,700.00         |  4209   |   4390                  |

</div>

<i> $\Delta$ (LGBM, Baseline) is the difference between models. </i>

As seen above the LGBM model can provide much better results in comparison to the baseline model, with an annual financial result around 172% better for 20,000 calls and 89% better for 40,000 calls, which is exactly what was shown in the Lift Curve. 

# 8. **Propensity Score List and Model Deployment**

<p align="justify"> The full list sorted by propensity score is available for download <a href="https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare_list.xlsx">here</a>. However, for other new future customers it was necessary to deploy the model. In this project Google Sheets and Render Cloud were chosen for that matter. The idea behind this is to facilitate the predictions access for any new given data, as those can be checked from anywhere and from any electronic device, as long as internet connection is available. The spreadsheet will return you the sorted propensity score for each client in the requested dataset, all you have to do is click on the "Propensity Score" button, then on "Get Prediction".

<b> Click here to access the spreadsheet </b>[![Sheets](https://www.google.com/images/about/sheets-icon.svg)](https://docs.google.com/spreadsheets/d/1K2tJP6mVJwux4qret1Dde9gQ23KsDRGRl8eJbsigwic/edit?usp=sharing)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the spreadsheet to provide a response, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>

# 9. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> We managed to provide a list of new customers ordered by their buy propensity score and a spreadsheet that returns the buy propensity score for other new future customers. Now, the Sales Team can focus their attention on the 20,000 or 40,000 first customers on the list, and in the future focus on the top K customers of the new list. </b> In addition to that, five interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Insuricare, as well as Expected Financial Results. </p>
 
# 10. **Next Steps**
<p align="justify"> Further on, this solution could be improved by a few strategies:
  
 - Conducting more market researches, so that more useful information on customers could be collected, since there was a lack of meaningful variables.
  
 - Applying <a href="https://builtin.com/data-science/step-step-explanation-principal-component-analysis">Principal Component Analysis (PCA) </a> in the dataset.
  
 - Try other classification algorithms that could better capture the phenomenon.

# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)
