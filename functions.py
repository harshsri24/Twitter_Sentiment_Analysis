import re
from nltk.corpus import stopwords


def stemming(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(tokens)



import pandas as pd

def preprocessing_data(word_query, number_of_tweets=50, use_tags = False):
    # Load tweet data from CSV
    tweet_data = pd.read_csv("test_data.csv")

    # Filter data based on the word query
    if use_tags:
        filtered_data = tweet_data[tweet_data['tweet'].str.contains('#' + word_query, case=False)]
    else :
        filtered_data = tweet_data[tweet_data['tweet'].str.contains(word_query, case=False)]

    # If the number of tweets requested is greater than available data, adjust it
    if number_of_tweets > len(filtered_data):
        number_of_tweets = len(filtered_data)

    # Prepare the DataFrame with required columns
    data = pd.DataFrame({
        'Tweets': filtered_data['tweet'].head(number_of_tweets),
        'target': filtered_data['label'].head(number_of_tweets),
    })
    return data