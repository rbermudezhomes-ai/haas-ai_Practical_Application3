# Practical Application III: Comparing Classifiers
Berkeley Haas Projects: Practical Application 3 [January 29, 2026]

### Goal

In this practical application, our goal is to compare the performance of each supervised machine learning algorithms - **Classification Algorithms** namely **K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines**.  We will utilize a dataset related to marketing bank products over the telephone. 


### Jupyter Notebook
[Practical_Application_3.ipynb](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/Practical_Application_3.ipynb)





### Directories and Files 
1. **data folder** contains:
- [bank-additional-full.csv](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/data/bank-additional-full.csv) - all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]

- [bank-additional.csv](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/data/bank-additional.csv) - 10% of the examples (4119), randomly selected from 1), and 20 inputs

2. **documents folder** contains:
- [CRISP-DM-BANK.pdf](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/documents/CRISP-DM-BANK.pdf) - research paper on 'Using Data Mining for Bank Direct Marketing'

- [bank-additional-names.txt](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/documents/bank-additional-names.txt) - contains citation and dataset relevant information

- [prompt_III_blank.ipynb](https://github.com/rbermudezhomes-ai/haas-ai_Practical_Application3/blob/main/documents/prompt_III_blank.ipynb) - blank Jupyter notebook template for the practical application


### Dataset Information

The dataset comes from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). 




**Variable Information**
```
Input variables:

# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):

21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
```


  
### Summary and Findings

The data is related with direct marketing campaigns of a Portuguese banking institution. There were 17 marketing campaigns and were based on phone calls. Often, more than one contact to the same client was required. During these phone campaigns, an attractive long-term deposit application, with good interest rates, was offered. These campaigns resulted in a total of 79,354 contacts with 6499 successes (an 8% success rate).  

Our main goal is to build a predictive model using classification algorithms to find out which existing customers are most likely to say "yes" or "no" to subscribe on bank term deposit subscription. Another objective is to improve the bank's marketing efficiency by identifying features that lead to higher conversion rates. These factors will allow the bank to reduce wasted effort on mass phone calls and marketing spend by targeting and identifying customers that are likely to subscribe.


**Model Comparison with Default Settings**

- Logistic Regression, KNN and Decision Tree are very fast to train. SVM has the slowest training time.
- Train Accuracy scores for Logistic Regression, KKN and SVM are close to each other while Decision Tree is the highest.
- Test Accurary scores for Logistic Regression, KKN and SVM are close to each other while Decision Tree is the lowest.
- Decision Tree has the highest Train Accuracy (99%) basically it memorized the training data and has lowest Test Accuracy, suggesting it is overfitting.
- Logistic Regression and SVM are the two models that are showing excellent generalization, both gaps are tiny. These scores suggest these models perform well on unseen data.
- Logistic Regression is the best model among the 4 models by just using the default settings. Logistic Regression took 0.4 seconds to get 89% accuracy while SVM took the longest time(> 4 minutes) among the 4 models to get 89% accuracy.

**Hyperparameters Fine Tuning with GridSearch**

```
"Logistic Regression": {'classifier__C': [0.1, 1, 10]},
"KNN": {'classifier__n_neighbors':list(range(1, 13, 2))},
"Decision Tree": {'classifier__max_depth': [None, 5, 10, 15, 20]}
"SVM": {'classifier__kernel': ['rbf'], 'classifier__gamma': ['scale', 0.01], 'classifier__C': [0.1, 1] }
```
To address the class imbalance in the dataset, we implemented __class_weight='balanced'__ during the hyperparameter tuning phase to prevent the model from being biased toward the majority class. This ensures the model prioritizes Recall over Accuracy. And as a result of the trade-off, the Accuracy will likely to drop, but it enhances Recall. This also ensures the model to be optimized for sensitivity within the minority class, where the cost of missed detection is highest.

Model Performace after Tuning: 

- Logistic regression is clearly the best model for this dataset. It has the highest Recall(64.5%) and the lowest training time(7.28 sec).
  We can tell the bank that we can use Logistic regression, a fast model that can catches the most "Yes" term subscribers(64.5%)
- KNN only looks at 1 neighbor and much likely it is just memorizing the training data(99.5% accuracy). This suggest an overfitting. This model is not a good choice for this dataset.
- Decision Treee is reliabe with a Recall(63%) but the tree max_depth:5 is shallow.
- SVM chooses a low values for gamma: 0.01 and C: 01, meaning the model chooses a simple boundary shape. The Recall performance (64%) is so close to Logistic Regression but the extra training time( over 20-40 minutes) for 2 CV-Folds is computationaly excessive.


### Recommendations and Next Step

These are the top 5 strongest features that we should focus on:
<br>
#### The Strongest Predictors

- cons.price.idx:  Strongest predictor. We can relate as the consumer price index rises, subscriptions increases. This suggests that customers are looking for stable bank products. 

- euribor3m: Interest rates have a positive pull on the outcome.

- pdays_contacted: Previous contact history is significant but in moderate volume.

- month_aug / mar: Specific months provides seasonal advantage.

<br>
These are the top 5 negative features that we should avoid:  
<br>

#### The Strongest Barriers
- emp.var.rate : This is the biggest barrier. High employment rate strongly suggest that customers are less likely to subsribe.

- month_may: May has the lowest conversion compared to the other months, even though it has the most volume call.

- contact_telephone: Calling people on their landlines is less successfull in this digital age.

- campaign: Suggesting an increased client calls(spamming the customer), the likely lead to saying No. 


#### Demographic Insights
Positive Drivers: Being Single , a Student , or Retired have a higher chance of saying Yes to term deposit.

Negative Drivers: Blue-collar workers , Divorced, and Unemployed  are less likely to subscribe.


<br>
The cons.price.idx and emp.var.rate are the 2 main dominant features - positive and negative respectively. This means the subscription is extremely sensitive to the economy. When Employment Variation Rate is high(uncertainty), we need to scale back on cold calling as they will likely to fail. Instead, during stable or high Consumer Price Index, increase on marketing resource as the data shows that customers are more inclined to subsribe on long term deposits.

Multiple and constant campaign calls to the same person have a strong negative effect and draws customer away from saying Yes, they are getting annoyed. A recommended approach is to manage minimum calls a week or a month and a planned calling schedule.





### Citation
This dataset is publicly available for research. The details are described in [Moro et al., 2014]. 
Please include this citation if you plan to use this database:

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
