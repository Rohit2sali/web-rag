from sentence_transformers import CrossEncoder

class reranking:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def re_rank_documents(self, query, documents):
        scores = self.model.predict([(query, doc) for doc in documents])

        ranked_documents = sorted(zip(documents, scores), key=lambda x : x[1], reverse=True)

        top_documents = [doc for doc, score in ranked_documents[:2]]

        return top_documents
