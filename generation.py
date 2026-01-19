from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from retriver import Hybrid_Search
from llama_index.llms.mistralai import MistralAI
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import BaseSynthesizer
import os

load_dotenv()

class prompt_template_generation():
    def __init__(self):
        self.search = Hybrid_Search()
        self.prompt_str = """You are an expert assistant.

        Use ONLY the information provided in the context below to answer the user question.
        If the answer is not present in the context, say:
        "I don’t know based on the provided context."

        Do NOT use prior knowledge.
        Do NOT hallucinate.

        Context:
        {context_str}

        User Question:
        {query_str}

        Answer in a clear, concise, and factual manner."""
        self.promt_tmpl = PromptTemplate(self.prompt_str)

    def prompt_generation(self, query):
        results = self.search.query_hybrid_search(query)

        context = "/n/n".join(results)

        promt_template = self.prompt_str.format(context_str=context, query_str=query)

        return promt_template
    

class RAGStringQueryEngine(CustomQueryEngine):
    llm : MistralAI
    response_synthesizer : BaseSynthesizer
    def custom_query(self, prompt):
        response = self.llm.complete(prompt)
        print("this is the direct response :",response)
        summary = self.response_synthesizer.get_response(query_str = str(response), text_chunks=str(prompt))

        return str(summary)

# @st.cache_resource
def create_query_engine(prompt):
    llm = MistralAI(model = "mistral-large-latest", api_key=os.environ.get("MISTRAL_API_KEY"))
    response_synthesizer = TreeSummarize(llm=llm)

    query_engine = RAGStringQueryEngine(llm=llm, response_synthesizer=response_synthesizer)

    response = query_engine.query(prompt)

    return response.response
