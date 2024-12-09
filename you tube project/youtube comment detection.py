import pandas as pd
import re
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# example YouTube comments
comments = [
    "I love this video! ",
    "bad content",
    "amazing video quality",
    "Amazing effort! Keep up the great work!",
    "I am not like the audio quality.",
    "this is the best video I have seen",
    "Check out this amazing content: https://youtu.be/Gx5qb1uHss4?si=CcFl6YwxH0lhl1Ie"
]

# Function perform kia
def clean_text(text):
    """
    Clean the input text by removing URLs, non-alphabetic characters, and converting to lowercase.
    """
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    return text

def detect_sentiment(text):
    """
    Detect the sentiment of the text using TextBlob.
    Returns: Positive, Neutral, or Negative.
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Main function
def main():
    print("Original Comments:")
    for comment in comments:
        print(f"- {comment}")
    print("\n")

    # Clean comment
    cleaned_comments = [clean_text(comment) for comment in comments]

    # Detect comment
    sentiments = [detect_sentiment(comment) for comment in cleaned_comments]

    
    print("Sentiment Analysis Results:")
    for original, sentiment in zip(comments, sentiments):
        print(f"Comment: {original}\nSentiment: {sentiment}\n")
    print("Visualizing Sentiment Distribution...")
    sns.countplot(x=sentiments, palette='coolwarm')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

# Run the script
if __name__ == "__main__":
    main()
