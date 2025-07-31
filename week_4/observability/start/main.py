import os
import glob
import openai
from pinecone import Pinecone
from typing import List, Dict, Any
from langsmith import Client, traceable
import time

# Initialize OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize Pinecone client
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pc = Pinecone(api_key=pinecone_api_key)

# LangSmith client setup
langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
langsmith_project = "rag-observability"
if langsmith_api_key:
    langsmith_client = Client(api_key=langsmith_api_key)
    print("LangSmith client initialized successfully")
else:
    print("Warning: LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled.")

# Constants
INDEX_NAME = "observability-test"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini-2024-07-18"

@traceable(name="load_documents")
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

@traceable(name="chunk_documents")
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

@traceable(name="get_embeddings")
def get_embeddings(texts: List[str]):
    """Generate embeddings for a list of texts using OpenAI."""
    response = openai.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [embedding.embedding for embedding in response.data]

@traceable(name="embed_documents")
def embed_documents(chunks: List[Dict[str, Any]], namespace: str) -> None:
    """
    Embed documents and store them in Pinecone.
    
    Args:
        chunks: List of document chunks with content and metadata
        namespace: Pinecone namespace to store the vectors
    """
    if not chunks:
        print("No chunks to process")
        return
        
    # Get Pinecone index
    try:
        index = pc.Index(INDEX_NAME)
        print(f"Connected to Pinecone index: {INDEX_NAME}")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index dimensions: {stats.dimension}")
        print(f"Total vectors: {stats.total_vector_count}")
        
        # Prepare batches (Pinecone usually works well with batches of ~100)
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            chunk_batch = chunks[i:i + batch_size]
            
            print(f"\nProcessing batch {batch_num}/{total_batches} with {len(chunk_batch)} chunks")
            
            # Get text from each chunk
            texts = [chunk["content"] for chunk in chunk_batch]
            
            # Get embeddings with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"  Getting embeddings (attempt {attempt + 1}/{max_retries})...")
                    embeddings = get_embeddings(texts)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"  Failed to get embeddings after {max_retries} attempts: {e}")
                        return
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Prepare data for Pinecone
            vectors = []
            for j, embedding in enumerate(embeddings):
                vector_id = f"chunk_{i + j}"
                metadata = chunk_batch[j].get("metadata", {})
                metadata["source"] = metadata.get("source", "unknown")
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone with retry logic
            for attempt in range(max_retries):
                try:
                    print(f"  Upserting {len(vectors)} vectors to namespace '{namespace}'...")
                    result = index.upsert(vectors=vectors, namespace=namespace)
                    print(f"  Successfully upserted batch {batch_num}/{total_batches}")
                    if hasattr(result, 'upserted_count'):
                        print(f"  Upserted {result.upserted_count} vectors")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"  Failed to upsert after {max_retries} attempts: {e}")
                        return
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    except Exception as e:
        print(f"Error in embed_documents: {e}")
        raise

@traceable(name="search_documents")
def search_documents(query, namespace, top_k=5):
    """Search the vector store with the user query."""
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Connect to Pinecone index
    index = pc.Index(INDEX_NAME, 
    host="observability-test-gyip7p4.svc.aped-4627-b74a.pinecone.io")

    # Verify the index connection
    try:
        stats = index.describe_index_stats()
        print(f"Successfully connected to index: {INDEX_NAME}")
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Error connecting to index: {e}")
        print("Please verify your API key and endpoint URL.")

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=False
    )
    
    # Get matched documents
    docs_with_scores = []
    for match in results["matches"]:
        # Load the document content based on the source file
        with open(match["metadata"]["source"], 'r') as f:
            content = f.read()
        docs_with_scores.append((content, match["score"]))
    
    return docs_with_scores

# TODO: Add traceable decorator to track this function in LangSmith
# Example: https://docs.smith.langchain.com/observability/how_to_guides/log_traces_to_project

@traceable(
    run_type="llm",
    name="ask_openai",
    project_name=langsmith_project
)
def ask_openai(query, documents, max_retries=3, initial_backoff=1):
    """
    Ask OpenAI a question with context from the documents.
    
    Args:
        query: The user's question
        documents: List of (document_text, score) tuples
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
    """
    # Sort documents by score (highest first) and take top 3
    sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)[:3]
    
    # Join top documents with a separator
    context = "\n---\n".join([doc for doc, _ in sorted_docs])
    
    # Truncate context to avoid hitting token limits
    max_context_length = 8000  # Conservative limit to stay under token limits
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [content truncated]"
    
    # Create messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that answers questions about Berkshire Hathaway "
                      "using provided shareholder letters. Be concise and accurate in your responses."
        },
        {
            "role": "user",
            "content": f"Context from shareholder letters:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    # Implement retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=500    # Limit response length
            )
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
                
            # Exponential backoff
            backoff_time = initial_backoff * (2 ** attempt)
            print(f"Rate limit hit. Retrying in {backoff_time} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff_time)
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            if attempt == max_retries - 1:
                return "I'm sorry, I encountered an error while processing your request. Please try again later."
            time.sleep(initial_backoff * (2 ** attempt))

def create_or_get_index():
    """Ensure the Pinecone index exists and return it."""
    try:
        # Get the correct dimension for the embedding model
        embedding_dimension = 1536  # text-embedding-3-small uses 1536 dimensions
        
        # List all indexes
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if INDEX_NAME in index_names:
            # Check if we need to recreate the index
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            if stats.dimension != embedding_dimension:
                print(f"Existing index has wrong dimension ({stats.dimension}). Deleting and recreating...")
                pc.delete_index(INDEX_NAME)
                # Wait for deletion to complete
                time.sleep(10)
                return create_or_get_index()
            return index
        
        # Create new index if it doesn't exist or was deleted
        print(f"Creating new index: {INDEX_NAME} with dimension {embedding_dimension}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-west-2"}}
        )
        print(f"Index {INDEX_NAME} created successfully")
        # Wait for index to be ready
        time.sleep(10)
        return pc.Index(INDEX_NAME)
        
    except Exception as e:
        print(f"Error creating/getting index: {e}")
        raise

if __name__ == "__main__":
    try:
        # Step 1: Initialize Pinecone index
        print("Initializing Pinecone...")
        index = create_or_get_index()
        
        # Step 2: Load and prepare documents
        print("\nLoading documents...")
        documents = load_documents()
        print(f"Loaded {len(documents)} documents")
        
        if not documents:
            print("No documents found. Please check the 'letters' directory.")
            exit(1)
        
        # Step 3: Chunk documents
        print("\nChunking documents...")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Step 4: Embed and store in Pinecone
        print("\nEmbedding and storing documents in Pinecone...")
        embed_documents(chunks, namespace="chunks")
        
        # Step 5: Search with a query
        user_query = "When did Berkshire Hathaway purchase its first Coke stock?"  # Year: 1988
        print(f"\nSearching for: {user_query}")
        
        docs_and_scores = search_documents(query=user_query, namespace="chunks")
        
        if not docs_and_scores:
            print("No matching documents found.")
            exit(1)
            
        print(f"\nFound {len(docs_and_scores)} relevant documents")
        
        # Step 6: Get answer from OpenAI
        print("\nGenerating response...")
        response = ask_openai(user_query, docs_and_scores)
        
        print("\n" + "="*80)
        print("QUESTION:", user_query)
        print("="*80)
        print("\nANSWER:")
        print(response)
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise