# dry-bean-prediction
Implementing dry bean classification using different models
## a. Problem Statement

In this project, I am trying to identify the type of dry bean from its given physical measurements. Different machine learning classification algorithms are trained using the bean features and their performance is compared. After training, I use a separate test dataset to check how well each model predicts the bean class. I have also created a Streamlit application where the trained models can be selected and their results can be viewed easily.


## b. Dataset Description

For this project, I used the Dry Bean Dataset. It contains measurements of individual dry beans, where each row represents one bean and the columns contain its physical characteristics.

The dataset contains **13,611 rows**, **16 input features**, and **7 different bean classes**. The classes are:

- Seker
- Barbunya
- Bombay
- Cali
- Dermosan
- Horoz
- Sira

The features describe different properties of the beans, such as area, perimeter, major axis length, minor axis length, aspect ratio and other shape-related measurements. The target column in my project is **`Class`**, which identifies the type of bean.

Before training the models, I removed duplicate rows and checked for missing or invalid values. I then divided the data into **80% training data and 20% testing data** using a stratified split.



## c. GitHub Repository Link

**GitHub Repository:** https://github.com/thaarunkumar001/dry-bean-prediction/


## d. Models Used and Evaluation Metrics

I trained five different classification models and compared their performance using Accuracy, AUC, Precision, Recall, F1 Score and MCC.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Random Forest | 0.9192 | 0.9919 | 0.9192 | 0.9192 | 0.9191 | 0.9022 |


## e. Observation on Performance

### Logistic Regression

Logistic Regression performed very well on this dataset. It reached an Accuracy of **0.9192** and the highest AUC of **0.9934**. Its MCC was also **0.9023**, showing that its predictions were quite reliable across the different classes.

### Decision Tree

The Decision Tree gave an Accuracy of **0.8955**, which was lower than Logistic Regression, KNN and Random Forest. Its MCC was **0.8737**. It performed reasonably well but was not the strongest model in this comparison.

### KNN

KNN achieved an Accuracy of **0.9155**, which is close to the top-performing models. Its AUC was **0.9811** and MCC was **0.8978**. This shows that KNN was also able to classify the bean types effectively.

### Gaussian Naive Bayes

Gaussian Naive Bayes had the lowest Accuracy at **0.7630** and the lowest MCC at **0.7143**. Its overall performance was noticeably lower than the other models.

### Random Forest

Random Forest achieved an Accuracy of **0.9192**, matching Logistic Regression for the highest Accuracy. Its MCC of **0.9022** was also almost the same as Logistic Regression. Its AUC was **0.9919**, slightly below Logistic Regression.

### Overall Observation

From my results, **Logistic Regression and Random Forest were the strongest models overall**. Logistic Regression had the highest AUC and MCC, while Random Forest had the same Accuracy. KNN was also close to these two models. Decision Tree performed reasonably well, while Gaussian Naive Bayes had the weakest results among the five models.

