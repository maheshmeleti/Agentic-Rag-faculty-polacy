import os
import warnings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

from src.logger import get_logger
from utils.common_functions import read_yaml
from src.agent.tool_utils import tool_name, tool_description
warnings.filterwarnings("ignore")


class RAGToolCreator:
    """
    A class to create a Retrieval-Augmented Generation (RAG) tool from a vector database.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAGToolCreator with configuration.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Initializing RAGToolCreator...")
        self.config = read_yaml(config_path)
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.logger.debug("RAGToolCreator initialized successfully.")
    
    def create_tool(self, db_path: str, tool_name: str, tool_description: str):
        """
        Create a RAG tool from the vector database.
        """
        self.logger.debug(f"Loading vector store from {db_path}...")
        vector_store = FAISS.load_local(
            db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(
            search_type=self.config["retrieval"]["search_type"], search_kwargs={"k": self.config["retrieval"]["top_k"]}
        )
        self.logger.debug("Vector store loaded successfully.")

        self.logger.debug(f"Creating retriever tool: {tool_name}...")
        retriever_tool = create_retriever_tool(
            retriever=retriever, name=tool_name, description=tool_description
        )
        self.logger.info(f"Retriever tool '{tool_name}' created successfully.")
        return retriever_tool

    def _initialize_llm(self) -> ChatOllama:
        """
        Initialize the language model (LLM) based on the configuration.
        """
        self.logger.debug("Initializing LLM...")
        model = self.config["llm"]["model"]
        base_url = self.config["llm"]["base_url"]
        llm = ChatOllama(model=model, base_url=base_url)
        self.logger.debug(f"LLM initialized with model: {model}.")
        return llm

    def _initialize_embeddings(self) -> OllamaEmbeddings:
        """
        Initialize the embeddings model based on the configuration.
        """
        self.logger.debug("Initializing embeddings...")
        model = self.config["embeddings"]["model"]
        base_url = self.config["embeddings"]["base_url"]
        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        self.logger.debug(f"Embeddings initialized with model: {model}.")
        return embeddings



if __name__ == "__main__":
    # Initialize the RAGToolCreator
    creator = RAGToolCreator(config_path="config.yaml")
    config = read_yaml('config.yaml')
    db_name_ = '2018-2019'
    db_path = os.path.join(config['data']['vector_db_path'], db_name_)

    # Create the RAG tool
    tool = creator.create_tool(db_path, tool_name(db_name_), tool_description(tool_name(db_name_)))

    # Test the tool with a sample query
    query = "What are the policies on faculty promotions for assistant professor?"
    response = tool.invoke(query)
    print(f"Query: {query}")
    print(f"Response: {response}")


