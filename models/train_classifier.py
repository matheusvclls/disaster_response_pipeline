import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import re


def load_data(database_filepath):
    """
    Loads the SQL database and generates categories labels, target and predcitor variables
    INPUT: Database's path
    OUTPUT: Categories labels, target and predcitor variables
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    tables_name = re.search("([/])\w+([.])",database_filepath)
    tables_name = tables_name.group(0)
    tables_name = tables_name[1:-1]
    df = pd.read_sql("SELECT * from "+ tables_name, engine)
    X = df['message']
    Y = df.drop(['message','original', 'id', 'genre'], axis=1)
    category_names = df.drop(labels=['id','message','original','genre'],axis=1).columns
    return X, Y, category_names
    
    
def tokenize(text):
    """
    Responsable for transforming the text in low, removes special characters ans removes stops words
    INPUT: Text of columns
    OUTPUT: Text of columns tokenized
    """

    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text).lower().strip()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words("english")]
    
    return text  


def build_model():
    """
    Responsable for building the model of machine learning. We have here the pipeline and the variables that will be tested
    in de gridsearch
    INPUT: None
    OUTPUT: model of machine learning
    """
        
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])    

    parameters = {
        'clf__estimator__n_estimators': [50]
        }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Responsable for evaluating the model, with the function of classification_report of skitlearn
    INPUT: model(gridsearch), X_test, Y_test and categories names
    OUTPUT: None
    """
    
    predicted = model.predict(X_test)
    
    for ind, cat in enumerate(category_names):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], predicted[ind]))
    

def save_model(model, model_filepath):
    """
    Saves the model
    INPUT: model and model_filepath
    OUTPUT: None
    """
 
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    """
    Run all functions above in a organized way
    """  
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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