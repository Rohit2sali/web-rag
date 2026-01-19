from dotenv import load_dotenv
import logging
from tqdm import tqdm
from qdrant_client import models
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from embedding import get_embedding_model

qdrant_client = QdrantClient(
    url = "http://localhost:6333"
)

print(qdrant_client.get_collections())

load_dotenv()
Qdrant_URL = "http://localhost:6333"
Collection_Name = "new_rag_docs"




class QdrantIndexing:
    def __init__(self, nodes):
        self.nodes = nodes
        self.embedding_model = get_embedding_model()
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL, 
        )
        self.metadata = []
        self.documents = []
        logging.info("Qdrant mdoel initialized .")

    def load_nodes(self):
        for node in self.nodes:
            self.metadata.append(node.metadata)
            self.documents.append(node.text)
        
        logging.info(f"loaded {len(self.nodes)} no of nodes from documents.")

    def client_collection(self):
        if not self.qdrant_client.collection_exists(collection_name=f"{Collection_Name}"):
            self.qdrant_client.create_collection(
                collection_name=Collection_Name,
                vectors_config={
                    'dense' : models.VectorParams(
                        size = 384,
                        distance = models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    'sparse' : models.SparseVectorParams(
                        index = models.SparseIndexParams(on_disk=False),
                    )
                }
            )
            logging.info(f"create {Collection_Name} in qdrant vector database.")

    def create_sparse_vector(self, text):
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]

        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("the embeddings does not have indices and values")
        
    def documents_insertion(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            dense_embedding = list(self.embedding_model.embed([doc]))[0]
            sparse_vector = self.create_sparse_vector(doc)

            point = models.PointStruct(
                id = i,
                vector={
                    'dense' : dense_embedding.tolist(),
                    'sparse' : sparse_vector,
                },
                payload = {
                    'text' : doc,
                    **metadata
                }
            )
            points.append(point)

        self.qdrant_client.upsert(
            collection_name=Collection_Name,
            points=points
        )

        logging.info(f"upserted {len(points)} points with dense and sparse vectors into Qdrant vectors database.")
