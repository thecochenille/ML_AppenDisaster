Machine Learning Project\
Appen Disaster Response
=============================================
# Project Summary: 
In the aftermath of disasters, it is challenging for organizations to rapidly make decisions on the type of response and emergency support to provide to victims. Using data science and machine learning can help speed up the decision process help responders provide the adequate response. 

In order to build a Machine Learning model that can help classify messages received from various media, we used the Appen Disaster Response dataset, an open-source project containing real world messages from various disasters from the past few years.

We used Natural Language Processing to build a Machine Learning model and made it available for use by building a Flask app, a Python web framework that was then deployed on XXX.

# The Dataset

The Appen Disaster Response dataset contains over 26,000 emergency messages classified into 36 response categories. Each message is also categorized by three genres, which is the origin of the message: as a direct message, a message from social media, or a message from a news outlet. You can visualize the distribution of genre at XXX.

# Project building process:

The project was built in three main steps:
- Data preparation: 
- Model building
- Flask App integration and deployment on 


# How to use the app
You can either run the app locally or go to XXX which allows you to have a general view of the dataset (we created a bar plot of messages by genre and wordclouds of the messages for each genre).


# How to run the scripts 
## Requirements
All the Python requirements are saved in requirements.txt

## If you want to run the scripts from the beginning
1. To prepare the data and cleaning it up, then saving the dataframe to an SQL database, run process_data.py in your Terminal as follows:
   
`python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`

2. Then, to train the model, run the train_classifier.py as follows:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

This will train and save your model to classifier.pkl

3. Once you have you model saved, you can use it with the Flask app and run it.

`cd app
python main.py`



# Repository files and folders:

├── app:
    │ └── templates
    │  ├── go.html # classification result page of web app
    │  ├── master.html # main page of web app
    │  
    └── main.py # Flask file that runs app

├── data:
   └── process_data.py #file contain the script to create ETL pipeline 
   └── disaster_categories.csv # data to process
   └── disaster_messages.csv # data to process 
   └── DisasterResponse.db   # database to save clean data to

├── models:
      └── train_classifier.py #file contain the script to create ML pipeline
	  └── classifier.pkl # saved model after running train_classifier.py

├── ETL Pipeline Preparation.ipynb

├── ML Pipeline Preparation.ipynb

├── README.md

├── requirements.txt


## jupyter notebooks
- ETL pipeline Preparation.ipynb: data preparation and cleaning
- ML pipeline Preparation.ipynb: model building and evaluation
- EDA pipeline Preparation.ipynb: data exploration and visualization




# Credit: 
- Dataset:


