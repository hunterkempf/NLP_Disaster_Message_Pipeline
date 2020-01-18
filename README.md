# Disaster Response Pipeline Project

### Motivation: 

In this project, I will apply NLP and Data Pipeline practices I have been learning to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data sets contain real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

My project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display several basic visualizations of the data.

### Basic Code Structure: 

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

### Instructions to Run on your Computer:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
