## Package for Web App
import streamlit as st
import joblib, os

## Package for Classification
import string, re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

## Package for WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add More Stopwords
stop = ['rp','bj','aku','radar','habibie','indonesia','jokowi','polisi','anak']

# Customized Preprocessing Functions
def preprocess_title(titles,more_stop=stop):

    # 1. Lowercase and Remove Punctuation
    lowercase_titles = titles.lower()
    no_pun = re.sub(r'[>)}:{",?+!\[\].(<;1234567890]','',lowercase_titles)
    no_pun = re.sub('\n','',no_pun)
    
    # 2. Remove Stopwords
    stopwords = StopWordRemoverFactory().get_stop_words() + more_stop
    data_stop = ArrayDictionary(stopwords)
    srf = StopWordRemover(data_stop)
    passed = srf.remove(no_pun)

    # 3. Stemming Process
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = stemmer.stem(passed)
        
    return stemmed

## Load Vectorizer
vectorizer = open('classification models/cv.pkl', 'rb')
cv = joblib.load(vectorizer)

## Prediction Function
def load_model(model,vect_text):
    fixed_model = joblib.load(open(os.getcwd() + '\\classification models\\' + model, 'rb'))
    proba = fixed_model.predict_proba(vect_text)
    if proba[0][1] > proba[0][0]:
        st.success('Judul Berita Clickbait! Nilai Probabilitas hingga {:.2f}%'.format(proba[0][1]*100))
    else:
        st.success('Judul Berita Clickbait! Nilai Probabilitas hingga {:.2f}%'.format(proba[0][0]*100))
    return fixed_model

def main():
    # Clickbait News App
    st.title('Aplikasi NLP dan ML untuk Bahasa Indonesia')
    st.subheader('Pilih aktivitas yang ingin dilakukan pada sidebar')

    activity = ['Klasifikasi Judul Clickbait', 'Visualisasi Word Cloud']
    task = st.sidebar.selectbox('Pilih Aktivitas', activity)
    st.sidebar.subheader('Tentang Aplikasi')
    st.sidebar.markdown('Aplikasi ini dibuat sebagai contoh penerapan NLP dan ML untuk Bahasa Indonesia menggunakan Python')
    st.sidebar.markdown('''Dibuat Oleh:  
    **_Raka Andriawan_**  
    **_Reynaldy Aries Ariyanto_**''')
    
    if task == 'Klasifikasi Judul Clickbait':
        st.info('Klasifikasi Judul Berita Clickbait dengan akurasi hingga 83%')
        title = st.text_area('Tulis Judul Berita','Ketik Disini')
        model_ml = ['Naive Bayes','Random Forest','Support Vector Machine']
        model_chosen = st.selectbox('Pilih Model ML', model_ml)
        
        if st.button('Klasifikasi'):
            vect_title = cv.transform([title])
            if model_chosen == 'Naive Bayes':
                estimator = load_model('mnb_model.pkl',vect_title)
            if model_chosen == 'Random Forest':
                estimator = load_model('rf_model.pkl',vect_title)
            if model_chosen == 'Support Vector Machine':
                estimator = load_model('svm_model.pkl',vect_title)
    
    if task == 'Visualisasi Word Cloud':
        st.info('Word Cloud dengan Stopword dan Stemmer untuk Bahasa Indonesia')
        article = st.text_area('Tulis Teks','Ketik Disini')
        stopword = st.text_area('Tambah Kata Stop','Ketik Disini')
        
        if st.button('Visualisasi'):
            stopword = stopword.split(' ')
            clean_article = preprocess_title(article,stopword)
            text_cloud = WordCloud(background_color='white',colormap='plasma').generate(clean_article)
            plt.imshow(text_cloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    
if __name__ == '__main__':
    main()