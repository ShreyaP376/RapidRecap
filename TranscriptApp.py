from flask import Flask, request
from youtube_transcript_api import YouTubeTranscriptApi
from summarizer import Summarizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from langdetect import detect

application = Flask(__name__)

@application.get('/summary')
def summary_api():
    
    url = request.args.get('url', '')
    max_length = int(request.args.get('max_length', 150))
    video_id = url.split('=')[1]

    try:
        transcript = get_transcript(video_id)
    except:
        return "No subtitles available for this video", 404

    # Extractive summarization using LSA or Frequency-based method
    if len(transcript.split()) > 3000:
        summary = extractive_summarization(transcript)
    else:
        summary = abstractive_summarization(transcript, max_length)

    return summary, 200

def is_transcript_english(transcript):
   
    try:
        language = detect(transcript)
        return language == 'en'
    
    except Exception as e:
        return False


def get_transcript(video_id):
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise e

    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript

def abstractive_summarization(transcript, max_length):
    
    summarizer = Summarizer()

    # Summarize the entire text
    summary = summarizer(transcript, max_length=max_length)

    return summary

def extractive_summarization(transcript):

    sentences = sent_tokenize(transcript)
    
    # Vectorize sentences
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    # Perform Truncated SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=1, random_state=42)
    svd.fit(X)
    components = svd.transform(X)
    
    # Rank sentences based on the first singular vector
    ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
    
    # Select top sentences for summary
    num_sentences = int(0.4 * len(sentences))  # 20% of the original sentences
    selected_sentences = sorted(ranked_sentences[:num_sentences])
    
    # Compile the final summary
    summary = " ".join([sentences[idx] for idx in selected_sentences])
    return summary


if __name__ == '__main__':
    application.run(debug=True)
