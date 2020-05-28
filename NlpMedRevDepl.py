import re, nltk
import time
import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


def normalizer(noRating):
    soup = BeautifulSoup(noRating, 'lxml')   # removing HTML encoding such as ‘&amp’,’&quot’
    souped = soup.get_text()
    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)"," ", souped) # removing @mentions, hashtags, urls

    tokens = nltk.word_tokenize(only_words)
    removed_letters = [word for word in tokens if len(word)>2]
    lower_case = [l.lower() for l in removed_letters]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def main():
    #### Loading the saved model
    model = joblib.load('svc.sav')
    vocabulary_model = pd.read_csv('ReviewsVocabulary_SVC.csv', header=None)
    vocabulary_model_dict = {}
    for i, word in enumerate(vocabulary_model[0]):
         vocabulary_model_dict[word] = i
    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary = vocabulary_model_dict, min_df=5, norm='l2', ngram_range=(1,3)) # min_df=5 is clever way of feature engineering
    #### Retrieving Review for user query
    #### Reading retrieved Review as dataframe
    noRating_df = pd.read_csv('NoRatings.csv' , encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Normalizing retrieved Review
    noRating_df['normalized_noRating'] = noRating_df.Review.apply(normalizer)
    noRating_df = noRating_df[noRating_df['normalized_noRating'].map(len) > 0] # removing rows with normalized Review of length 0
    print("Number of Review remaining after cleaning: ", noRating_df.normalized_noRating.shape[0])
    print(noRating_df[['Review','normalized_noRating']].head())
    #### Saving cleaned Review to csv file
    noRating_df.drop(['Medicine','Condition', 'Review'], axis=1, inplace=True)
    noRating_df.to_csv('cleaned_reviews.csv', encoding='utf-8', index=False)
    cleaned_reviews = pd.read_csv("cleaned_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    cleaned_reviews_tfidf = tfidf.fit_transform(cleaned_reviews['normalized_noRating'])
    targets_pred = model.predict(cleaned_reviews_tfidf)
    #### Saving predicted sentiment of Review to csv
    cleaned_reviews['predicted_rating'] = targets_pred.reshape(-1,1)
    cleaned_reviews.to_csv('predicted_rating.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()
