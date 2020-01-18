# import libraries
import sys
import pickle

from sqlalchemy import create_engine
import pandas as pd
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Using the database filepath this function will pull data from the
    Message table in the sqlite db that was stored from the Data Engineering
    process earlier that cleaned and joined the data. It will return the X,Y
    and category names of the target columns
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Message",con=engine)
    X = df.message
    Y = df.loc[:, "related":"direct_report"]
    category_names = Y.columns
    return X, Y, category_names 


def tokenize(text):
    """
    This function is a tokenizer function that is used by CountVectorizer
    it removes URLs and replaces them with a placeholder to have a more consistent
    text feature. It also tokenizes and Lemmatizes the words in the message.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    This is a transformer that creates features related to if the first word
    of a sentance is a verb
    """
    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    This function builds the sklearn pipeline to create text based features
    and classify messages into each of the 36 target columns
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
    
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
    
            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 25)))
    
    ])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    This function takes in a trained model and the test data (features and actuals) and uses
    the model to predict results. It then runs those results through a classification report for
    each of the 36 target columns.
    """
    y_pred = model.predict(X_test)
    for colnum in range(y_pred.shape[1]):
        print("\nClassification Report for",category_names[colnum],"Column")
        print(classification_report(y_test.values[:,colnum],y_pred[:,colnum]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 2:
        database_filepath = sys.argv[1]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Searching Hyper Parameters for Pipeline...')
        parameters = {'features__text_pipeline__tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [25,26],
              'clf__estimator__min_samples_split': [3, 4],
              'features__transformer_weights': (
                  {'text_pipeline': 1, 'starting_verb': 0.5},
                  {'text_pipeline': 1, 'starting_verb': 1}
              )
             }
        cv = GridSearchCV(model,parameters,verbose=2)
        cv.fit(X_train, Y_train)
        print("Best Parameters Found...")
        print(cv.best_params_)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
