import sys
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle
import os



def load_data(database_filepath):
    '''
    this function loads the data from the database created in process_data.py, 
    and separates labels and features into Y and X.

    Argument: database filepath

    Output: X = dataframe of categories
            Y = dataframe of feature (here message contain texts)
            category_names = names of each column of Y

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('data', engine)
    X = df['message']
    Y = df.iloc[:,-36:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    this function takes an array text, prepares it by removing punctuations, 
    creating word tokens, and lemmatize and remove caps. the clean tokens are saved 
    in clean_tokens 

    Input: text = 

    Output: clean_tokens = 
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #removing all punctuations
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        #clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
        
    return clean_tokens



def build_model(X_train,Y_train):
    #define the pipeline
    print(X_train.head())
    pipeline = Pipeline([
            ('vect', CountVectorizer(token_pattern=None,tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #create a model with parameter tuning for clf using GridSearchCV
    #n_estimators = [10, 100, 1000]
    #criterion = ['gini', 'entropy', 'log_loss']
    #max_features = ['sqrt', 'log2', None] # we can also look at int and float values but I decided not to to reduce the number of searches for this exercise.

    #parameters = {'clf__estimator__n_estimators': n_estimators,
    #              'clf__estimator__criterion': criterion,
    #              'clf__estimator__max_features': max_features}

    
    #model = GridSearchCV(pipeline, param_grid = parameters)
    model = pipeline

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    #print('The best parameters are : ' + model.best_params_)
    #predicting Y from X_test using the best estimator from GridSearchCV
    Y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        print('Classification report for: ' + col)
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filename):
    app_directory = 'app/'
    filepath = os.path.join(app_directory, model_filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()