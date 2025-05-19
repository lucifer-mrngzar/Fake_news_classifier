# app.py 


import streamlit as st
import joblib


#loading model and the vectorizer

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#function to clean and predict 

def clean_text(text):
    import re
    import string 
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    text = text.lower()
    text = re.sub(r'\[.*?\]','', text)
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'<.*?>+','',text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*','',text)
    # now removing the stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text 

#deciding whether the news is fake or real 

def predict_news(text):

    cleaned_text = clean_text(text)
    
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    return 'Fake News' if prediction == 0 else 'Real News'

# streamlit web app layouts and features

st.title("Fake News Classifier")

st.subheader("Enter a news headline or paragraph and find out if it's Real or Fake.")

user_input = st.text_area("Your News Text")

if st.button("Classify"):
    if user_input.strip() != "":

        result = predict_news(user_input)
        st.success(f"Prediction : **{result}**")
    else :
        st.warning("Please enter some news text before clicking Classify.")
            



