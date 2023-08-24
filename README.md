Machine Learning Project\
Appen Disaster Response
=============================================
PROJECT STATUS (Aug 24 2023)
- Deploying the Flask app to AWS Elastic Beanstalk using Docker

=============================================
# Project Summary: 
In the aftermath of disasters, it is challenging for organizations to rapidly decide on the type of response and emergency support to provide victims. Data science and machine learning can help speed up the decision process and help responders provide adequate responses. 

To build a Machine Learning model that can help classify messages received from various media, we used the Appen Disaster Response dataset, an open-source project containing real-world messages from various disasters from the past few years.

We used Natural Language Processing to build a Machine Learning model and made it available by building a Flask app, a Python web framework.

# The Dataset

The Appen Disaster Response dataset contains over 26,000 emergency messages classified into 36 response categories. Each message is also categorized into three genres, which is the origin of the message: a direct message, a message from social media, or a message from a news outlet.

# Project building process:

The project was built in three main steps:
- Data preparation: 
- Model building
- Flask App integration and deployment on 


# How to use the app
You can run the app locally (How to run the scripts) which allows you to have a general view of the dataset (we created a bar plot of messages by genre and wordclouds of the messages for each genre).


# How to run the scripts 
## Requirements
All the Python requirements are saved in requirements.txt

## If you want to run the scripts from the beginning
1. To prepare the data and clean it up, then save the dataframe to an SQL database, run process_data.py in your Terminal as follows:
   
`python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`

2. Then, to train the model (no saved model in the repo), run the train_classifier.py as follows:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

This will train and save your model to classifier.pkl

3. Once you have your model saved, you can use it with the Flask app and run it locally.

`cd app
python main.py`


# What does it look like?
## Data visualization page
You can visualize the dataset used to train the machine learning model from a bar plot categorizing each message by their origin, and wordclouds showing what the most frequent words were in the different types of messages.

Below the data visualization, you can test the machine learning model by inserting an emergency message.

![](https://github.com/thecochenille/ML_AppenDisaster/blob/498729efe0d690ce2d08508d723f3086ba7ae846/ScreenShot1.png)

# Repository files and folders:
```bash
├── app:
    │ └── templates
    │  ├── go.html # classification result page of web app
    │  ├── master.html # main page of web app
    │  ├── template.html # template for the classification result page
    │  
    └── main.py # Flask file that runs app

├── data:
   └── process_data.py #file contain the script to create ETL pipeline 
   └── disaster_categories.csv # data to process
   └── disaster_messages.csv # data to process 
   └── DisasterResponse.db   # database that is used in models, was generated by process_data.py

├── models:
      └── train_classifier.py #file contain the script to create ML pipeline and train the train dataset. The script uses GridSearchCV to try different parameters of Random Forest so it takes a while to run. If you want to just run the model with defaults parameters, use train_classifier2.py 
	└── train_classifier2.py # this is the faster version of the machine learning pipeline, with defaults parameters, it will generate a pkl file to use in the app.
├── EDA.ipynb # this file contains the python script to create the data visualizations used on the Flask App

├── ETL Pipeline Preparation.ipynb

├── ML Pipeline Preparation.ipynb

├── README.md

├── requirements.txt #the list of Python packages used to work on this project

```



## jupyter notebooks
- ETL pipeline Preparation.ipynb: data preparation and cleaning
- ML pipeline Preparation.ipynb: model building and evaluation
- EDA.ipynb: data exploration and visualization




# Credits: 
- Dataset : [Appen](https://appen.com/) provided the Disaster Response dataset
- Code : [Udacity](www.udacity.com) provided the templates for the Flask App


