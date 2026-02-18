from transformers import pipeline

def get_sentiment_score(news_list):
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

    results = sentiment_pipeline(news_list)

    score_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    scores = [score_map[result['label'].lower()] for result in results]

    return sum(scores) / len(scores)


if __name__ == "__main__":
    news = [
        "Apple reports record profits",
        "Apple faces regulatory challenges"
    ]

    score = get_sentiment_score(news)
    print("Sentiment Score:", score)
