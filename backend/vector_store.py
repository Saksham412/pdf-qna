from transformers import AutoTokenizer, AutoModel
import torch
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv() 

api_key = os.getenv('PINECONE_API_KEY')
index = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")

model_name = os.getenv("EMBEDDING_MODEL")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


# pinecone.init(api_key=api_key, environment=pinecone_env)
# index = pinecone.Index(index_name)
pc = Pinecone(
    api_key=api_key
)

# Now do stuff
if index not in pc.list_indexes().names():
    pc.create_index(
        name=index, 
        dimension=1024, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

def store_embeddings(embeddings, metadata):
    for i, embedding in enumerate(embeddings):
        index.upsert([(f"doc-{metadata['file']}-{i}", embedding, metadata)])

def search_vector_db(query, top_k=5):
    query_embedding = embed_text(query)
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return results
