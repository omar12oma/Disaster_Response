import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from sqlalchemy import create_engine

# Tokenization function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data from SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('ETL', engine)

Y = df.drop(columns=['id','message','original','genre'])
top_10_category = Y.sum().sort_values(ascending=False)[:10]
top_10_category_index = list(top_10_category.index)

# Load model
model = joblib.load("../models/classifier.pkl")

# Streamlit app

# Sidebar for input
st.sidebar.header('Enter a disaster message:')
query = st.sidebar.text_area('Message Text', '')

# Display title and introduction
st.title('Disaster Response Classification App')
st.write("This web app classifies disaster messages into various categories using a pre-trained model.")

# Display the distribution of message genres
st.header("Distribution of Message Genres")
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

# Create genre distribution bar chart
fig1 = go.Figure([go.Bar(x=genre_names, y=genre_counts)])
fig1.update_layout(
    title="Distribution of Message Genres",
    xaxis_title="Genre",
    yaxis_title="Count"
)
st.plotly_chart(fig1)

# Display the top ten categories
st.header("Top Ten Categories")
fig2 = go.Figure([go.Bar(x=top_10_category_index, y=top_10_category)])
fig2.update_layout(
    title="Top Ten Categories",
    xaxis_title="Categories",
    yaxis_title="Count"
)
st.plotly_chart(fig2)

# Classification section
if query:
    st.header("Message Classification Result")
    
    # Use the model to predict the categories for the query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # Display the prediction results
    for category, label in classification_results.items():
        st.write(f"{category}: {'Yes' if label == 1 else 'No'}")

