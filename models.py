from nltk.corpus import stopwords
from wordcloud import WordCloud
import time

from imports import *
from functions import *

## ________________________________________________________________________________________________________________________________
## Logistic regression Setup

# print('Transforming data to numerical form for Logistic Regression ...')
vectorizer = pickle.load(open('logistic_vectorizer.pkl', 'rb'))

# Logistic regressor
# print('Loading Logistic regressor model  ...')
loaded_model :LogisticRegression = pickle.load(open("sentiment_analyser_model.sav", 'rb'))

def predict_by_logistic(sentence):
    preprocessed_sentence = stemming(sentence)
    numerical_sentence = vectorizer.transform([preprocessed_sentence])
    prediction = loaded_model.predict(numerical_sentence)
    return prediction
    
    

## ________________________________________________________________________________________________________________________________
## 
## LSTM Setup



tokenizer:Tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
# load model
# log_to('Loading LSTM model ...')
# print('reloaded')
model:Sequential = load_model('model_final0.h5')
# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_using_lstm(sentence):
    preprocessed_sentence = stemming(sentence)
    numerical_sentence = tokenizer.texts_to_sequences([preprocessed_sentence])
    numerical_sentence = pad_sequences(numerical_sentence, maxlen = 200)
    prediction = model.predict([numerical_sentence])[0]
    return prediction



# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white', 
                          colormap='viridis',  # Choose a color map (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis')
                        #   font_path='path_to_your_font_file.ttf',  # Specify the path to your font file
                          max_words=200,  # Maximum number of words to display
                          contour_color='steelblue',  # Color of word cloud outline
                          contour_width=2,  # Width of word cloud outline
                          stopwords=None).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Show word cloud in Streamlit
    st.pyplot(plt)




import pandas as pd

# Assume that preprocess_data returns a DataFrame with a column named 'Tweets'

def predict_by_logistic_df(data):
    preprocessed_tweets = data['Tweets'].apply(stemming)
    numerical_tweets = vectorizer.transform(preprocessed_tweets)
    predictions = loaded_model.predict(numerical_tweets)
    # Decode the predictions
    decoded_predictions = [decode(prediction) for prediction in predictions]
    data['Logistic Prediction'] = decoded_predictions
    data['Logistic Prediction Score'] = predictions
    return data

def predict_by_lstm_df(data):
    # Preprocess the 'Tweets' column of the DataFrame
    preprocessed_tweets = data['Tweets'].apply(stemming)
    # Convert the preprocessed tweets into numerical form using the tokenizer
    numerical_tweets = tokenizer.texts_to_sequences(preprocessed_tweets)
    numerical_tweets = pad_sequences(numerical_tweets, maxlen=200)
    predictions = model.predict(numerical_tweets)
    decoded_predictions = [decode(prediction) for prediction in predictions]
    # Add the predictions to the DataFrame
    data['LSTM_Prediction'] = decoded_predictions
    data['LSTM Prediction Score'] = predictions

    return data

### ________________________________________________________________________________________________________________________________
### Streamlit code
### 

import streamlit as st
st.set_page_config(layout="wide")

# Function to decode sentiment score
def decode(x):
    return 'Positive' if x > 0.5 else 'Negative'

# Streamlit app title with animation
st.title("🚀 Sentiment Analysis Demo")

# Create two columns for layout: one for radio buttons and one for input field
col1, col2 = st.columns([1, 2])

# Radio buttons for search options (on the left side) with animation
with col1:
    st.write("🔍 Search by:")
    search_option = st.radio("", ("Word/Hashtag", "User tweet"))

# Input field for search query (on the right side) with animation
with col2:
    if search_option == "Word/Hashtag":
        search_query = st.text_input("Enter Word/Hashtag to search:")
    else:
        search_query = st.text_input("Enter User tweet:")
    submitted = st.button("Analyze")

if submitted:
    st.write("🔍 Analyzing...")

    # Simulate processing delay with animation
    with st.spinner('Analyzing...'):
        st.write("✨ Analysis complete!")

    # Fetch data from file and analyze all of it
    if search_option == "Word/Hashtag":
        # Preprocess data based on search query
        data :pd.DataFrame = preprocessing_data(search_query)
        if data.empty == True:
            st.write("💂‍♂️ No tweets found!")
            # stop this bolck from further processing
            st.stop()
        prediction_logistic = predict_by_logistic_df(data)
        prediction_lstm = predict_by_lstm_df(data)

        # Display predictions with animated transitions
        st.subheader("Predictions table on the retrieved data")
        
        styled_prediction_logistic = prediction_logistic.style.applymap(lambda x: 'background-color: #7FFF00' if (x == 'Positive' or( type(x)==float and x >=0.5 ) or (type(x) == int and x == 1))else 'background-color: #FF6347', subset=pd.IndexSlice[:, prediction_logistic.columns != 'Tweets'])        
        
        
        st.table(styled_prediction_logistic)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Logistic Regression Prediction")
            st.bar_chart(prediction_logistic['Logistic Prediction'].value_counts(), color='#000000')

        # Bar chart for LSTM predictions
        with col2:
            st.subheader("LSTM Prediction")
            st.bar_chart(prediction_lstm['LSTM_Prediction'].value_counts(), color='#000000')
            
        # Generate word cloud from tweets with fade-in animation
        
                # Generate word cloud from tweets
        all_tweets_text = ' '.join(data['Tweets'])
        st.subheader("Word Cloud")
        generate_wordcloud(all_tweets_text)

    else:
        # predictions by both
        prediction_logistic = predict_by_logistic(search_query)
        prediction_lstm = predict_using_lstm(search_query)
        
        # Display predictions with animated transitions
        st.subheader("Predictions:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Logistic Regression Prediction:")
            st.write(f'{decode(prediction_logistic)} {prediction_logistic}')
            st.bar_chart(prediction_logistic, color='#000000')

        with col2:
            st.subheader("📈 LSTM Prediction:")
            st.write(f'{decode(prediction_lstm)} {prediction_lstm}')
            st.bar_chart(prediction_lstm, color='#000000')
