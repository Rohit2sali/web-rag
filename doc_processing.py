from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
import re
from indexing import QdrantIndexing

def indexing(nodes):
    indexing = QdrantIndexing(nodes=nodes)
    indexing.load_nodes()
    indexing.client_collection()
    indexing.documents_insertion()

def transoform(documents):
    transformed_documents = []
    for doc in documents:
        transformed_content = doc.get_content().lower()
        transformed_content = re.sub(r'\s+', ' ', transformed_content)
        transformed_content = re.sub(r'[^\w\s]', '', transformed_content)
        transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
    return transformed_documents

def sentence_splitter_doc_into_nodes(all_documents):
    try:
        splitter = SentenceSplitter(chunk_size=600, chunk_overlap=100)

        nodes = splitter.get_nodes_from_documents(all_documents)
        return nodes
    
    except Exception as e:
        print(f"Unable to split documents into nodes : {e}")
        return []

class Custom_transformation:
    def __init__(self, documents):
        self.document = documents
        transformed_docs = transoform(documents)
        nodes = sentence_splitter_doc_into_nodes(transformed_docs)
        indexing(nodes=nodes)
