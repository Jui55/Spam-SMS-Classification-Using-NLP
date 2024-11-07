# Spam Detection Project
## Project Overview
This project is dedicated to building a model that identifies spam messages within SMS text data. The goal is to develop a reliable classification tool that can filter out spam from legitimate messages using machine learning models.

## Source
https://www.kaggle.com/datasets/mariumfaheem666/spam-sms-classification-using-nlp

## Table of Contents
    Data Overview
    Data Visualization
    Data Preprocessing
    Model Training and Evaluation
    Model Comparison and Best Model Selection
    Files
    Requirements
    Conclusion


### Data Overview
The dataset includes SMS messages with labels that classify each message as either Spam or Not Spam. This classification allows us to train a model that can recognize patterns in spam content. Key features of the dataset:

    Text: The SMS content, which requires text processing for analysis.
    Label: Binary indicator (Spam or Not Spam) used as the target for model training.
      
### Data Visualization

#### Class Distribution:
Visualized using bar charts and pie charts to illustrate the balance between spam and ham messages.
#### Text Analysis:
Word clouds are generated for both spam and ham messages to observe frequently used terms within each class.
#### Message Length Analysis: 
Histograms are used to show the distribution of message lengths, helping understand length patterns in spam and ham messages.ry classification.

### Bar chart for class distribution
![image_alt](https://github.com/Jui55/Spam-SMS-Classification-Using-NLP/blob/main/Screenshot%202024-11-07%20213535.png)

### Pie chart for class distribution
![image_alt](https://github.com/Jui55/Spam-SMS-Classification-Using-NLP/blob/main/Screenshot%202024-11-07%20213608.png)

### WordCloud for spam and ham messages
![image_alt](https://github.com/Jui55/Spam-SMS-Classification-Using-NLP/blob/main/Screenshot%202024-11-07%20213631.png)
### Histogram for message lengths
![image_alt](https://github.com/Jui55/Spam-SMS-Classification-Using-NLP/blob/main/Screenshot%202024-11-07%20213507.png)


### Data Preprocessing

#### Text Vectorization: 
The TfidfVectorizer (Term Frequency-Inverse Document Frequency) is used to transform text data into numerical form. This process helps convert SMS messages into features usable by machine learning models.
#### Encoding Target Variable: 
The target column is mapped to numerical values for binary classification.

### 4.Model Training and Evaluation
Three different machine learning models are implemented, trained, and evaluated:

    1.Naive Bayes
    2.Logistic Regression
    3.Support Vector Machine (SVM)
#### Performance Evaluation
The models are evaluated using standard metrics, including:

    Accuracy: Proportion of correct predictions.
    Precision: Correctness of spam predictions.
    Recall: Completeness of spam prediction.
    F1-Score: Harmonic mean of precision and recall.
    Confusion Matrix: Visualization of true positives, true negatives, false positives, and false negatives.


   
### 5.Model Comparison and Best Model Selection
A comparison of model performances is printed, and the best model is selected based on accuracy.After comparing the performance of each model, Support Vector Machine was identified as the best model with an accuracy of 98%. This model demonstrated the highest accuracy among the three evaluated models (Naive Bayes, Logistic Regression, and Support Vector Machine), making it the optimal choice for classifying SMS messages as spam or ham.

## Files
      1.Spam_SMS.csv: The dataset containing SMS messages and their spam/ham labels.
      2.SpamFinal.ipynb: Jupyter Notebook containing code for the entire spam detection pipeline.
## Requirements
The following libraries are used in this project:

  #### pandas, numpy, matplotlib, seaborn:
  For data manipulation and visualization.
  #### sklearn:
  For model implementation, TF-IDF vectorization, and evaluation metrics.
  #### wordcloud: 
  For visualizing frequently occurring words in spam and ham messages.

## Conclusion
This project provides a full pipeline to classify SMS messages as spam or ham using popular machine learning models. Logistic Regression, Naive Bayes, and Support Vector Machine models are compared, with the best model identified for final use.
