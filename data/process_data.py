import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    categories = df['categories'].str.split(pat=';' ,expand=True)
    row = categories.iloc[0, :].str.split('-').values
    category_colnames = [c[0] for c in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories=categories[categories['related']!=2]

    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_index=True, right_index=True)

    # drop the duplicates
    df=df.drop_duplicates()

    return df


def save_data(df, database_filename):
    url = 'sqlite:///' + database_filename
    engine = create_engine(url)
    df.to_sql(name='ETL', con=engine, if_exists='replace', index=False)


def main():
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