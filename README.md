# Disaster Response Pipeline Project

### Motivation: 

In this project, I will apply NLP and Data Pipeline practices I have been learning to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data sets contain real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

My project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display several basic visualizations of the data.

### Basic Code Structure: 
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 
```

### Extract Transform Load (ETL): 

My ETL pipeline reads in the messages and categories CSV files, joins them on the id column, splits the categories into separate columns, removes duplicate rows and saves the clean dataset to a sqlite database. 

### Machine Learning Natural Language Processing Pipeline: 

##### High level view: 

At a very high level view the ML pipeline is processing the message data and outputting a prediction on each of 36 categories from if the message is related to a disaster to if the message is about requests for medical help to if the message is about children that are alone. 

message data --> Pipeline --> prediction on 36 categories 

The full list of 36 categories is shown below:

1. 'related'
1. 'request'
1. 'offer'
1. 'aid_related'
1. 'medical_help'
1. 'medical_products'
1. 'search_and_rescue'
1. 'security'
1. 'military'
1. 'child_alone'
1. 'water'
1. 'food'
1. 'shelter'
1. 'clothing'
1. 'money'
1. 'missing_people'
1. 'refugees'
1. 'death'
1. 'other_aid'
1. 'infrastructure_related'
1. 'transport'
1. 'buildings'
1. 'electricity'
1. 'tools'
1. 'hospitals'
1. 'shops'
1. 'aid_centers'
1. 'other_infrastructure'
1. 'weather_related'
1. 'floods'
1. 'storm'
1. 'fire'
1. 'earthquake'
1. 'cold'
1. 'other_weather'
1. 'direct_report'

##### Lower level view:
- Training/Prediction Pipeline
    - Feature Creation Pipeline
        - tfidf pipe
            - Count Vectorizer
            - TFIDF Transformer
        - Starting Verb Extractor
    - Model
        - Random Forest Classifier --> Multi-Output Classifier 

My Feature Creation Pipeline focuses on 2 main features Term Frequency Inverse Document Frequency (TFIDF) and if the first word of a sentance is a verb. This can and will be extended to more text based features which can help to improve the model's ability to predict categories especially rare event categories. 

My Model uses a simple Random Forest Classifier and extends it to handle the 36 targets using a Multi-Output Classifier which is a nice feature in scikit-learn to fit one classifier per target and keep code clean.

### Hyperparameter Tuning:

I did a grid search over a set of hyper parameters to tune my model using scikit-learn's GridSearchCV function over the entire pipeline. 

### Instructions to Run on your Computer:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database (you should delete data/DisasterResponse.db first)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves classifier model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
