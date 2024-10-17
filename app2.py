import streamlit as st
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from textblob import TextBlob
import openai
import io
import pandas as pd
from transformers import pipeline


# Load environment variables
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create client using endpoint and key
credential = AzureKeyCredential(ai_key)
ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def analyze_sentiment(text):
    # Analyzes sentiment using Azure Text Analytics
    response = ai_client.analyze_sentiment([text])[0]
    return response.sentiment, response.confidence_scores

def suggest_alterations(text):
    # Use TextBlob to provide basic suggestions
    blob = TextBlob(text)
    suggestions = []

    for sentence in blob.sentences:
        if sentence.sentiment.polarity < 0:  # Negative sentiment
            suggestions.append(f"Consider revising: {sentence}")

    return suggestions


    
    return response['choices'][0]['message']['content'].strip()

def summarize_text(text):
    # Generate a summary
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def summarize_large_text(text):
    # Split text into chunks (assuming 1000 characters per chunk)
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)


def main():
    st.title("Azure Text Analytics with Suggestions")

    # Text input options
    input_option = st.radio("Choose input method:", ("Enter Text", "Upload File"))

    if input_option == "Enter Text":
        text = st.text_area("Enter the text you want to analyze:", height=200)
    else:
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file is not None:
            text = io.TextIOWrapper(uploaded_file).read()
        else:
            text = ""

    if st.button("Analyze"):
        if text:
            try:
                st.write("### Original Text:")
                st.write(text)
                
                # Get language
                detectedLanguage = ai_client.detect_language(documents=[text])[0]
                st.subheader("Language Detection")
                st.write(f"Detected Language: {detectedLanguage.primary_language.name}")
                
                # Analyze text using Azure Text Analytics
                sentiment, confidence_scores = analyze_sentiment(text)
                st.write(f"### Sentiment Analysis:")
                st.write(f"Overall Sentiment: {sentiment}")
                st.write(f"Confidence Scores: {confidence_scores}")
                
                confidence_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Confidence': [
                        confidence_scores.positive,
                        confidence_scores.neutral,
                        confidence_scores.negative
                        ]
                    })
                st.bar_chart(confidence_df.set_index('Sentiment'))

                # Analyze each sentence individually
                blob = TextBlob(text)
                st.write("### Individual Sentence Sentiment Analysis:")
                for sentence in blob.sentences:
                    sentence_sentiment = sentence.sentiment
                    st.write(f"**Sentence:** {sentence}")
                    st.write(f"**Sentiment:** {'positive' if sentence_sentiment.polarity > 0 else 'negative' if sentence_sentiment.polarity < 0 else 'neutral'}")
                    st.write(f"**Confidence Scores:** {{'positive': {sentence_sentiment.polarity}, 'neutral': {1 - abs(sentence_sentiment.polarity)}, 'negative': {1 - sentence_sentiment.polarity}}}")

                # Show alteration suggestions for negative sentences
                suggestions = suggest_alterations(text)
                if suggestions:
                    st.write("### Suggestions for Improving Text:")
                    for suggestion in suggestions:
                        st.write(suggestion)
                
                # Generate summary
                if len(text) > 1000:
                    summary = summarize_large_text(text)
                else:
                    summary = summarize_text(text)

                st.write("### Summary:")
                st.write(summary)
                
            except Exception as ex:
                st.error(f"An error occurred: {str(ex)}")  
                
        else:
            st.warning("Please enter some text or upload a file to analyze.")
          

        # Optionally, show AI-generated suggestions (advanced)
                

       

if __name__ == '__main__':
    main()

