import sys
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' 
    load_data
    This function loads our messages and categories datasets 
    into messages and categories.

    Input: 
    messages_filepath : a file path to the message dataset (csv format)
    categories_filepath : a file path to the categories dataset (csv format)

    Output: 
    df : a dataframe where messages and categories are merged
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merging messages and categories into one dataframe.
    df = messages.merge(categories, on="id")
    
    return df


def clean_data(df):
    '''
    clean_data
    This function cleans up the dataset by:
    - creating names for labels and cleaning up categories columns
    - removing duplicate rows

    Input: 
    df : dataframe created by load_data

    Output:
    df : clean dataframe
    '''

    #creating labels names and cleaning up columns
    
    categories = df['categories'].str.split(';',expand=True)
    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    

    df=df.drop(columns=['categories'], axis=1)
    df = pd.concat([df,categories],axis=1)

    #removing duplicates
    df=df.drop_duplicates()

    return df



def save_data(df, database_filename):
    ''' 
    save_data
    This function creates a database called DisasterProject.db and 
    saves our clean dataset with a filename that you specify.

    Input:
    df : a clean dataframe
    database_filename: a filename to save the dataset into the database
    
    Output:
    none


    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Data all clean!')
     
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()