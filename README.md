#Text Analysis and Summarization App
This project is a web application built with Streamlit that integrates Azure Text Analytics, OpenAI, and Hugging Face Transformers. It allows users to analyze sentiment, detect languages, and generate summaries of the text. The app also provides suggestions for improving the text, especially for negative sentences.

#Features
Sentiment Analysis: Analyze the overall sentiment of a given text (Positive, Neutral, Negative) using Azure Text Analytics API.
Language Detection: Detect the primary language of the input text.
Sentence-Level Sentiment: Perform sentence-by-sentence sentiment analysis using TextBlob.
Text Suggestions: Suggest improvements for negative sentiment sentences using TextBlob.
Text Summarization: Summarize long text passages using a pre-trained summarization model from Hugging Face (DistilBART).
File Upload: Analyze text from either direct input or from an uploaded file.

# Screenshots
![image](https://github.com/user-attachments/assets/612726b7-92c1-431b-8337-2fb756db3303)
![image](https://github.com/user-attachments/assets/42a32727-b076-4e61-bcc1-9eee3e8bec3d)
![image](https://github.com/user-attachments/assets/547c6093-7f1e-4e70-b8ad-524782f67798)





## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. Create a virtual environment and activate it:
    ```
    python -m venv venv
     ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Set up the environment variables in a `.env` file:
    ```
    AI_SERVICE_ENDPOINT=<your_azure_text_analytics_endpoint>
    AI_SERVICE_KEY=<your_azure_text_analytics_key>
    OPENAI_API_KEY=<your_openai_key>
    ```

5. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

 #Usage
Enter Text: You can either manually input text in the provided text area or upload a text file for analysis.
Analyze: Click the "Analyze" button to perform sentiment analysis, language detection, and summarization.
View Results: The app displays the following:
Detected language of the text.
Sentiment analysis with confidence scores.
Sentence-by-sentence sentiment analysis.
Suggestions for revising negative sentences.
Summary of the text.

#Dependencies
Streamlit: Frontend for the web application.
Azure AI Text Analytics: For sentiment analysis and language detection.
TextBlob: For sentence-level sentiment analysis and suggestions.
Hugging Face Transformers: For text summarization using a pre-trained model (DistilBART).
OpenAI: (Optional) For more advanced AI-generated suggestions.

#Acknowledgements
Azure Text Analytics API
TextBlob
Hugging Face Transformers
Streamlit

