# Disaster_Response

The project involves analyzing disaster data from Appen to build a model for an API that classifies disaster messages. The dataset contains real messages sent during disaster events. The goal is to create a machine learning pipeline to categorize these messages so they can be sent to appropriate disaster relief agencies.

## Installation

```python
pip install pandas
pip install numpy
pip install nltk
pip install plotly
pip install flask
pip install sqlalchemy
pip install scikit-learn


```
## Instructions to run 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Testing the model
![water](https://github.com/omar12oma/Disaster_Response/assets/129009511/b36668aa-213c-4e57-8590-f514c3dc8f0f)
