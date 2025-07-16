from openai import OpenAI
import faiss
import numpy 

# Example documents about movies
summaries = [
    "The Godfather is a classic mafia crime drama",
    "Inception explores dreams within dreams",
    "The Shawshank Redemption is a story about hope and friendship",
]

def get_embedding(text):
    # TODO: Implement the OpenAI embedding functionality
    # Documentation: https://platform.openai.com/docs/guides/embeddings
    # 1. Create an OpenAI client

    client = OpenAI()

    # 2. Make an API call to generate embeddings using the text-embedding-3-small model

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    # 3. Return the embedding vector from the response
    # Placeholder for the actual implementation
    # return [0] * 1536  # Placeholder with the expected dimension

    return response.data[0].embedding

embeddings = []
for summary in summaries:
    embedding = get_embedding(summary)
    embeddings.append(embedding)


dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

index.add(numpy.array(embeddings, dtype='float32'))

query = 'Tell me about a prison movie'

query_embedding = get_embedding(query)
distances, indicies = index.search(numpy.array([query_embedding], dtype='float32'), 3)

for i in range(3):
    print(f"Match {i+1}, Distance: {distances[0][i]}")
    print(summaries[indicies[0][i]]) 