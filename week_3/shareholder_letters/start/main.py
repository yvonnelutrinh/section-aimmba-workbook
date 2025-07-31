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
INDEX_NAME = "letters-test"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini-2024-07-18"

# Get API key from .zshrc or environment
pinecone_api_key = os.environ.get("PINECONE_API_KEY") or get_api_key_from_zshrc()
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your environment or .zshrc file.")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the index
index = pc.Index(INDEX_NAME, host="letters-test-gyip7p4.svc.aped-4627-b74a.pinecone.io")

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
    """
    Split documents into meaningful chunks while preserving context.
    Tries to split at paragraph boundaries when possible.
    """
    chunks = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        
        # First, split into paragraphs (double newlines)
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and (current_length + len(para) > chunk_size):
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": metadata
                })
                
                # Start new chunk with overlap from previous chunk
                overlap_start = max(0, len(current_chunk) - chunk_overlap // 100)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(p) for p in current_chunk) + 2 * len(current_chunk)  # +2 for newlines
            
            current_chunk.append(para)
            current_length += len(para) + 2  # +2 for newlines
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
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
        for j, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
            # Include both the original metadata and the content in the vector's metadata
            vector_metadata = chunk["metadata"].copy()
            vector_metadata["content"] = chunk["content"]  # Store the actual text content
            
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": embedding,
                "metadata": vector_metadata
            })
        
        # Upsert to Pinecone
        try:
            upsert_response = index.upsert(vectors=vectors, namespace=namespace)
            print(f"Upserted batch {i//batch_size + 1}, response: {upsert_response}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
            raise

def search_documents(query, namespace, top_k=5):
    """Search the vector store with the user query."""
    print(f"\nSearching for: {query}")
    
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Search Pinecone
    result = index.query(
        namespace=namespace,
        vector=query_embedding, 
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    # Process and print results
    print(f"\nTop {top_k} matches:")
    docs_with_scores = []
    if hasattr(result, 'matches'):
        for i, match in enumerate(result.matches, 1):
            doc_text = match.metadata.get('content', '') if hasattr(match, 'metadata') else ''
            # Show first 100 chars of each match
            preview = (doc_text[:100] + '...') if len(doc_text) > 100 else doc_text
            print(f"\nMatch {i} (Score: {match.score:.3f}):")
            print(f"Preview: {preview}")
            docs_with_scores.append((doc_text, match.score))
    
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

def clear_index():
    """Delete all vectors from the index."""
    global index
    try:
        # First get all vector IDs
        stats = index.describe_index_stats()
        if stats['namespaces'] and 'chunks' in stats['namespaces']:
            # Delete all vectors in the 'chunks' namespace
            index.delete(delete_all=True, namespace="chunks")
            print("Successfully cleared all vectors from the 'chunks' namespace")
        else:
            print("No vectors found to delete")
    except Exception as e:
        print(f"Error clearing index: {e}")

def is_index_populated():
    """Check if the index already has documents."""
    try:
        stats = index.describe_index_stats()
        return stats['namespaces'].get('chunks', {}).get('vector_count', 0) > 0
    except Exception as e:
        print(f"Error checking index status: {e}")
        return False

if __name__ == "__main__":
    # Check if index is already populated
    if not is_index_populated():
        print("Index is empty. Loading and embedding documents...")
        docs = load_documents()
        chunks = chunk_documents(docs)
        embed_documents(chunks, namespace="chunks")
        print("Documents embedded successfully!")
    else:
        print("Index already contains documents. Using existing index.")

    # Step 2: Write a query
    user_query = "When did Berkshire Hathaway purchase it's first coke stock?" # Year: 1988

    # Step 3: Check Pinecone for similar chunks
    docs_and_scores = search_documents(query=user_query, namespace="chunks")
    for _, score in docs_and_scores:
        print(f"Score: {score}")
    
    # Step 4: Put docs into prompt and send to OpenAI
    response = ask_openai(user_query, docs_and_scores)
    print(response) 