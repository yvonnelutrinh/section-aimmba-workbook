from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """

    # TODO: Create a prompt that:
    # 1. Asks for sentiment analysis
    # 2. Specifies the required output format
    #       thought: [analysis]
    #       sentiment: [positive/negative]
    # 3. Includes the review text

    prompt = f"""
    Given the review provided, analyze the sentiment of the review as positive or negative, and provide a general analysis of the review as a thought.
    
    The review is: {review}

    The output should be in the format: 
        thought: [analysis] 
        sentiment: [positive/negative]
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents= prompt,
    )

    content = response.text

    # read the response format by printing
    # print(content)

    content_parts = content.split("sentiment:")
    thought = content_parts[0].strip().replace("thought: ", "")
    sentiment = content_parts[1].strip()

    # TODO: Parse the response to extract thought and sentiment
    # The response should be in the format:
    # thought: [analysis]
    # sentiment: [positive/negative]
    
    result = {
        "thought": thought,
        "sentiment": sentiment
    }
    
    return result

def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 1):
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    main() 