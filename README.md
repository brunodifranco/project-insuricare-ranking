<h1 align="center"> Creating a bot that predicts Rossmann future sales</h1>

<p align="center">A learning to rank cross-sell project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/195202224-01bfd468-9f1c-4e83-af60-b101312a98e3.svg" alt="drawing" width="800"/>
</p>

*Obs: Business problem, company and data are fictitious.*

*The in-depth Python code explanation is available in [this](https://github.com/brunodifranco/project-rossmann-sales/blob/main/rossmann.ipynb) Jupyter Notebook.*

# 1. **Insuricare and Business Problem**
<p align="justify"> Insuricare is an insurance company that has provided health insurance to its customers, and now they are willing to sell a new vehicle insurance to their clients. To achieve that, Insuricare conducted a research with around 305 thousand customers that bought the health insurance last year, asking each one if they would be interested in buying the new insurance. This data was stored in the company's database, alongside other customers' features. 

Then, Insuricare Sales Team selected around 76 thousand new customers, which are people that didn't respond to the research, to offer the new vehicle insurance. However, due to a limit call <i>restriction*</i> Insuricare must choose a way of selecting which clients to call: </p>

- Either select the customers randomly, which is our <b>baseline model</b>.
  
- Or, the Data Science Team will provide, by using a Machine Learning (ML) model, an ordered list of these new customers, based on their propensity score of buying the new insurance.

<i> * Insuricare Sales Team would like to make 20,000 calls, but it can be pushed to 40,000 calls. </i>

# 2. **Data Overview**
The training data was collected from a PostgreSQL Database, while the test data is available at [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction/code). The initial features descriptions are available below:

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

- Cross-selling is strategy used to sell products associated with another product already owned by the customer. In this project, health insurance and vehicle insurance are the products. 
- Learning to rank is a machine learning application, in order to rank a variable. In this project, we are ranking customers in a list, from the most likely customer to buy the new insurance to the least likely one. This list will be provided by the ML model.

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To provide an ordered list of these new customers, based on their propensity score of buying the new insurance the following steps were performed: </p>

- <b> Understanding the Business Problem </b> : Understanding the main objective Insuricare was trying to achieve and plan the solution to it. 

- <b> Collecting Data </b>: Collecting data from a PostgreSQL Database, as well as from Kaggle.

- <b> Data Cleaning </b>: Checking data types and Nan's. Other tasks such as: renaming columns, dealing with outliers, changing data types weren't necessary at this point. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA) </b>: Exploring the data in order to obtain business experience, look for useful business insights and find important features for the ML model. The top business insights found are available at <a href="https://github.com/brunodifranco/project-rossmann-sales#5-top-business-insights"> Section 4 </a>. </p>

- <b> Feature Engineering </b>: Editing original features, so that those could be used in the ML model. 

- <b> Data Preparation </b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/"> Rescaling Techniques</a> in the data, as well as <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Enconding Methods</a>, to deal with categorical variables. 

- <b> Feature Selection </b>: Selecting the best features to use in the ML model by using <a href="https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f"> Random Forest </a>. 

- <p align="justify"> <b> Machine Learning Modeling </b>: Training Classificaion Algorithms with cross-validation. The best model was selected to be improved via Bayesian Optimization with Optuna. More information at <a href="https://github.com/brunodifranco/project-rossmann-sales#6-machine-learning-models">Section 5 </a>. </p>

- <b> Model Evaluation </b>: Evaluating the model using two metrics: Precision at K and Recall at K, as well as two curves: Cumulative Gains and Lift Curves. 

- <b> Results </b>: Translating the ML model's to financial and business performance.

- <p align="justify"> <b> Model Deployment </b>: Providing the ranked customers, alonside a propensity score, in Google Sheets. This is the project's <b> Data Science Product </b>, and it can be accessed from anywhere. More information at <a href="https://github.com/brunodifranco/project-rossmann-sales#7-model-deployment"> Section 6 </a>. </p>
  
## 4.2. Tools and techniques used:
- [Python 3.10.8](https://www.python.org/downloads/release/python-3108/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Sklearn](https://scikit-learn.org/stable/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [Python API's](https://realpython.com/api-integration-in-python/).  
- [Render Cloud](https://render.com/) and [Google Sheets](https://www.google.com/sheets/about/).
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
  <img src="https://user-images.githubusercontent.com/66283452/198151788-1018458c-8e67-4ead-9d15-76622b4df287.png" alt="drawing" width="850"/>
</p>

--- 

- ### 3rd - Men are more likely to buy the new vehicle insurance than women.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198151862-4de5cab2-1647-4aae-bc9a-b0772e91ef18.png" alt="drawing" width="850"/>
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

The initial performance for all seven algorithms are displayed below (ordered by Precision at K):

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|    CatBoost Classifier   | 0.3099 +/- 0.0011  |0.8274 +/- 0.003 |
|    AdaBoost Classifier   | 0.3098 +/- 0.0018  |0.8273 +/- 0.0049|
|      LGBM Classifier     | 0.3075 +/- 0.0015  |0.8209 +/- 0.004 | 
|    Logistic Regression   | 0.3058 +/- 0.0012  |0.8165 +/- 0.0033|
|      XGB Classifier      | 0.2992 +/- 0.0018  |0.7988 +/- 0.0049|
| Random Forest Classifier | 0.2949 +/- 0.0014  |0.7874 +/- 0.0037|
|      KNN Classifier      | 0.2739 +/- 0.003   |0.7314 +/- 0.0081|
</div>

<i>K here is either equal to 20,000 or 40,000, given our business problem. </i>

<p align="justify"> The <b>Light GBM Classifier</b> model will be chosen for hyperparameter tuning, since it's by far the fastest algorithm to train and tune, with similar results to CatBoost and AdaBoost. </p>

LGBM speed in comparison to other ensemble algorithms trained in this dataset:
- 4.7 times faster than CatBoost 
- 7.1 times faster than XGBoost
- 30.6 times faster than AdaBoost
- 63.2 times faster than Random Forest

<p align="justify"> At first glance the models performances don't look so great, and that's due to short amount of variables, on which many are categorical or binary, or simply those don't contain much information. 

However, <b>for this business problem</b> this isn't a major concern, since the goal here isn't finding the best possible prediction on whether a customer will buy the new insurance or not, but to <b>create a score that ranks clients in a ordered list, so that the sales team can contact them in order to sell the new vehicle insurance</b>.</p>

After tuning LGBM's hyperparameters using [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) the model performance has improved on the Precision at K, and decreased on Recall at K, which was expected: 

<div align="center">

|         **Model**        | **Precision at K** | **Recall at K** |
|:------------------------:|:------------------:|:---------------:|
|      LGBM Classifier     |       0.33320  	|     0.72000 	  | 

</div>

## <i>Metrics Definition and Interpretation</i>

<p align="justify"> <i> As we're ranking customers in a list, there's no need to look into the more traditional classification metrics, such as accuracy, precision, recall, f1-score, aoc-roc curve, confusion matrix, etc.

Instead, **ranking metrics** will be used:

- **Precision at K** : Shows the fraction of correct predictions made until K out of all predictions. 
  
- **Recall at K** : Shows the fraction of correct predictions made until K out of all true examples. 

In addition two curves can be plotted: 

- <b>Cumulative Gains Curve</b>, indicating the percentage of customers, ordered by probability score, containing a percentage of all customers interested in the new insurance. 

- <b>Lift Curve</b>, which indicates how many times the ML model is better than the baseline model (original model used by Insuricare). </i> </p>

# 7. **Business and Financial Results**

## 7.1. Business Results

**1) By making 20,000 calls how many interested customers can Insuricare reach with the new model?**
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152035-48c27ead-53f8-440e-af92-f049456dac33.png" alt="drawing" width="1000"/>
</p>

<p align="justify"> 

- 20,000 calls represents 26.24% of our database. So if the Sales Tteam were to make all these calls Insuricare would be able to contact 72.30% of customers interest in the new vehicle insurance, since 0.7230 is our recall at 20,000. </p>

- As seen from the Lift Curve, our **LGBM model is 2.75 times better than the baseline model at 20,000 calls.** 

**2) Now increasing the amount of calls to 40,000 how many interested customers can Insuricare reach with the new model?**

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152040-929e3f17-d07e-401a-892c-50bf9c01f475.png" alt="drawing" width="1000"/>
</p>

- 40,000 calls represents 52.48% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 99.48% of customers interest in the new vehicle insurance, since 0.9948 is our recall at 40,000. 

- At 40,000 calls, our **LGBM model is around 1.89 times better than the baseline model.**  

# 7. **Model Deployment**

<p align="justify">  As previously mentioned, the complete financial results can be consulted by using the Telegram Bot. The idea behind this is to facilitate the access of any store sales prediction, as those can be checked from anywhere and from any electronic device, as long as internet connection is available.  
The bot will return you a sales prediction over the next six weeks for any available store, <b> all you have to do is send him the store number in this format "/store_number" (e.g. /12, /23, /41, etc) </b>. If a store number if non existent the message "Store not available" will be returned, and if you provide a text that isn't a number the bot will ask you to enter a valid store id. 

To link to chat with the Rossmann Bot is [![image](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rossmann_project_api_bot)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the bot to respond, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>

# 8. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> A model that can provide good sales predictions for each store over the next six weeks was successfully trained and deployed in a Telegram Bot, which fulfilled CEO' s requirement, for now it's possible to determine the best resource allocation for each store renovation. </b> In addition to that, five interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Rossmann CEO. </p>
 
# 9. **Next Steps**
<p align="justify"> Further on, this solution could be improved by a few strategies:

 - Using <a href="https://towardsdatascience.com/an-introduction-to-time-series-analysis-with-arima-a8b9c9a961fb">ARIMA</a> to predict the amount of customers over the next six weeks, so that the customers column could be added to the final model. </p>
 
 - Tune even more the regression algorithm, by applying a <a href="https://machinelearningmastery.com/what-is-bayesian-optimization/">Bayesian Optimization</a> for instance. 
  
 - Try other regression algorithms to predict the sales for each store.
 
 - Use different models for the stores on which it's more difficult (higher MAE and MAPE) to predict the sales.

# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)
