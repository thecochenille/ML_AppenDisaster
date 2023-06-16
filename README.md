Machine Learning Project\
Appen Disaster Response
=============================================
# Project Summary: 
This project uses the Appen Disaster Response dataset to build a Machine Learning model and to classify messages to 36 emergency response categories.

# Project building process:
The project was built in three main steps:
- Data preparation
- Model building
- Flask App integration (in progress)




# How to use the app


# How to run the scripts 
## Requirements

## Execution
1. To prepare the data and cleaning it up, then saving the dataframe to an SQL database, run process_data.py as follows:
   
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

2. Then, to train the model, run the train_classifier.py as follows:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl



# Repository files and folders:
## data/

## models/
## app/
- run.py: script running the app with data viz and use the model

## jupyter notebooks
- ETL pipeline Preparation.ipynb: data preparation and cleaning
- ML pipeline Preparation.ipynb: model building and evaluation
- EDA pipeline Preparation.ipynb: data exploration and visualization




# Credit: 
- Dataset:


