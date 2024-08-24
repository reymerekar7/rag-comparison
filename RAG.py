
import pandas as pd
import nest_asyncio
# from dotenv import load_dotenv
from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import PromptTemplate

import tempfile
import os

from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


## based on new streamlit logic, might have to modify the classes here to return the query engine as well as the index, such that we store it in the streamlit session

class CohereRAGPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.query_engine = None
        self.index = None

    def handle_file_upload(self, uploaded_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            return temp_dir, file_path

    def initialize_models(self):
        llm = Cohere(api_key=self.api_key, model="command-r-plus")

        embed_model = CohereEmbedding(
            cohere_api_key=self.api_key,
            model_name="embed-english-v3.0",
            input_type="search_query",
        )

        cohere_rerank = CohereRerank(
            model='rerank-english-v3.0',
            api_key=self.api_key,
        )

        return llm, embed_model, cohere_rerank

    def load_data_and_create_index(self, temp_dir, embed_model):
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            required_exts=[".pdf"],
            recursive=True
        )
        docs = loader.load_data()
        Settings.embed_model = embed_model
        
        self.index = VectorStoreIndex.from_documents(docs, show_progress=True)
        return self.index

    def create_query_engine(self, llm, cohere_rerank):
        Settings.llm = llm
        self.query_engine = self.index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        return self.query_engine

    def get_query_engine(self):
        return self.query_engine

    def get_index(self):
        return self.index
    

# llm = OpenAI(model = "gpt-4-turbo")
# embed_model = OpenAIEmbedding(model="text-embedding-3-large")

class OpenAIRAGPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.query_engine = None
        self.index = None

    def initialize_models(self):
        llm = OpenAI(model = "gpt-4-turbo", api_key=self.api_key)
 
        # default is text-embedding-ada-002
        embed_model = OpenAIEmbedding(api_key=self.api_key)

        return llm, embed_model
    
    def load_data_and_create_index(self, temp_dir, embed_model):
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            required_exts=[".pdf"],
            recursive=True
        )

        docs = loader.load_data()
        Settings.embed_model = embed_model

        self.index = VectorStoreIndex.from_documents(docs, show_progress=True)

        return self.index

    def create_query_engine(self, llm):
        Settings.llm = llm
        self.query_engine = self.index.as_query_engine(streaming=True)
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        return self.query_engine

    def get_query_engine(self):
        return self.query_engine

    def get_index(self):
        return self.index
    

# outside abstraction to create embeddings and index independent of the ones confined in the classes

def load_data_and_create_index(temp_dir, embed_model):
    loader = SimpleDirectoryReader(
        input_dir=temp_dir,
        required_exts=[".pdf"],
        recursive=True
    )
    docs = loader.load_data()
    Settings.embed_model = embed_model
    
    index_ = VectorStoreIndex.from_documents(docs, show_progress=True)
    return index_