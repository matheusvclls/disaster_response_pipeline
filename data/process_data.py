import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import re


def load_data(messages_filepath, categories_filepath):
    """
    Generates an unique dataset with information of messages and categories
    
    INPUT: Path of messages and categories dataset
    OUTPUT: Dataframe in pandas with both of these datasets
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = categories.merge(messages, how='left', on='id')
    
    return df

    
    
def clean_data(df):
    """
    Generates a dataset treated
    
    INPUT: Dataset with messages and categories without treatment
    OUTPUT: Dataset treated
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = [x[:-2] for x in categories.iloc[0].tolist()]
    category_colnames = row
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x) if x.isdigit() else 0) 

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)   
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")

    # remove all labels that are different of 0 and 1
    df = df[(df['related'] == 0) | (df['related'] == 1)]

    # drop duplicates
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Save the result in a database SQL
    
    INPUT: Dataframe treated and path of database to save
    OUTPUT: None
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    tables_name = re.search("([/])\w+([.])",database_filename)
    tables_name = tables_name.group(0)
    tables_name = tables_name[1:-1]
    df.to_sql(tables_name, engine, index=False)  


def main():
    """
    Run all functions above in a organized way
    """    

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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