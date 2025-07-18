import os
import glob
import openai
import re
from typing import List
from pinecone import Pinecone

def get_api_key_from_zshrc():
    """Read the Pinecone API key from .zshrc file."""
    zshrc_path = os.path.expanduser("~/.zshrc")
    try:
        with open(zshrc_path, 'r') as f:
            for line in f:
                if 'PINECONE_API_KEY' in line and not line.startswith('#'):
                    # Extract the API key using regex
                    match = re.search(r'export\s+PINECONE_API_KEY=["\']?([^\'\n\"]+)["\']?', line)
                    if match:
                        return match.group(1)
    except Exception as e:
        print(f"Error reading .zshrc: {e}")
    return None

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pinecone Serverless settings (update these for your project):
# See your Pinecone console for correct values
# Pinecone configuration
INDEX_NAME = "test"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini-2024-07-18"

# Get API key from .zshrc or environment
pinecone_api_key = os.environ.get("PINECONE_API_KEY") or get_api_key_from_zshrc()
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your environment or .zshrc file.")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the index
index = pc.Index(INDEX_NAME, host="test-gyip7p4.svc.aped-4627-b74a.pinecone.io")

# Verify the index connection
try:
    stats = index.describe_index_stats()
    print(f"Successfully connected to index: {INDEX_NAME}")
    print(f"Index stats: {stats}")
except Exception as e:
    print(f"Error connecting to index: {e}")
    print("Please verify your API key and endpoint URL.")

def load_documents():
    """Load all text documents from the letters directory."""
    documents = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "letters/*.txt")
    files = glob.glob(path)
    
    for file_path in files:
        with open(file_path, 'r') as file:
            content = file.read()
            documents.append({"content": content, "metadata": {"source": file_path}})
    
    print(f"Found {len(documents)} letters")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better processing."""
    chunks = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Simple text splitting - you can implement more sophisticated chunking if needed
        for i in range(0, len(content), chunk_size - chunk_overlap):
            if i > 0:
                start = i - chunk_overlap
            else:
                start = 0
                
            chunk_content = content[start:start + chunk_size]
            if chunk_content:
                chunks.append({"content": chunk_content, "metadata": metadata})
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single piece of text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=512  # Ensure we match the index dimension
    )
    return response.data[0].embedding

def get_embeddings(texts: List[str]):
    """Generate embeddings for a list of texts using OpenAI."""
    embeddings = [get_embedding(text) for text in texts]
    return embeddings

def embed_documents(chunks, namespace):
    """Embed documents and store them in Pinecone."""
    # Use the global index object (classic API)
    global index
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        
        # Get text from each chunk
        texts = [chunk["content"] for chunk in chunk_batch]
        
        # Get embeddings
        embeddings = get_embeddings(texts)
        
        # Prepare data for Pinecone
        vectors = []
        for j, embedding in enumerate(embeddings):
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": embedding,
                "metadata": chunk_batch[j]["metadata"]
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)

def search_documents(query, namespace, top_k=5):
    """Search the vector store with the user query."""
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # TODO: Search Pinecone using the query_embedding
    # Implement the search functionality using the Pinecone Index query method
    # Documentation: https://sdk.pinecone.io/python/pinecone/grpc.html#GRPCIndex.query
    # The query should:
    # 1. Get a reference to the index
    # 2. Call the query method with the appropriate parameters
    # 3. Process the results to extract the documents
    
    result=index.query(
        namespace=namespace,
        vector=query_embedding, 
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    # Placeholder for the actual implementation
    docs_with_scores = []
    return docs_with_scores

def ask_openai(query, documents):
    """Ask OpenAI a question with context from the documents."""
    # Join all documents into a single context string
    context = "\n\n".join([doc for doc, _ in documents])
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # # Step 1: Load document embeddings into Pinecone - only run this the first time
    # docs = load_documents()
    # chunks = chunk_documents(docs)
    # embed_documents(chunks, namespace="chunks")

    # Step 2: Write a query
    user_query = "When did Berkshire Hathaway purchase it's first coke stock?" # Year: 1988

    # Step 3: Check Pinecone for similar chunks
    docs_and_scores = search_documents(query=user_query, namespace="chunks")
    for _, score in docs_and_scores:
        print(f"Score: {score}")
    
    # Step 4: Put docs into prompt and send to OpenAI
    response = ask_openai(user_query, docs_and_scores)
    print(response) 