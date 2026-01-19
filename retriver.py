import logging 
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
import streamlit as st

load_dotenv()
Qdrant_URL = "http://localhost:6333"
Collection_Name = "new_rag_docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @st.cache_resource
class Hybrid_Search():
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            timeout=30
        )

    def query_hybrid_search(self, query, limit=10):

        dense_query = list(self.embedding_model.embed([query]))[0].tolist()

        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

        results = self.qdrant_client.query_points(
            collection_name=Collection_Name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                    using="sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
                ),
            limit=limit
        )

        documents = [point.payload["text"] for point in results.points]

        return documents
    
