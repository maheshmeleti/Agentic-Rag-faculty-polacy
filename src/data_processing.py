import os
import warnings
from typing import List, Dict, Any
import yaml

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.logger import get_logger
from utils.common_functions import read_yaml

warnings.filterwarnings("ignore")

class DocumentProcessor:
    """
    A class to handle the complete document processing pipeline from PDF loading to vector storage.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DocumentProcessor with configuration.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Initializing DocumentProcessor...")
        
        self.config = read_yaml(config_path)
        self.embeddings = self._initialize_embeddings()
        self.text_splitter = self._initialize_text_splitter()
        self.logger.debug("DocumentProcessor initialized successfully.")
    
    def process_and_store(self, name):
        """
        Go to every folder and process the documents and store
        """
        self.logger.info('Started processing folders...')
        # for name in os.listdir(self.config['data']['input_dir']):
        pdfs_path = os.path.join(self.config['data']['input_dir'], name)
        self.logger.info(f"Processing {name}...")
        db_name = name
        self.process_and_store_(pdfs_path, db_name)
        self.logger.info(f"Completed processing {name}")

    
    def process_and_store_(self, pdfs_path: str, db_name: str) -> None:
        """
        Complete pipeline to process PDFs and store as vector database.
        """
        self.logger.debug(f"Starting vectorization process for {pdfs_path}...")
        
        # Step 1: Find all PDF files
        pdf_paths = self._load_pdfs(pdfs_path)
        if not pdf_paths:
            self.logger.warning(f"No PDFs found in {pdfs_path}.")
            return
        
        # Step 2: Load and split documents
        chunks = self._process_documents(pdf_paths)
        
        # Step 3: Initialize vector store
        vector_store = self._initialize_vector_store()
        
        # Step 4: Add documents to vector store
        ids = vector_store.add_documents(documents=chunks)
        self.logger.debug(f"Added {len(ids)} documents to vector store. Total vectors: {vector_store.index.ntotal}.")
        
        # Step 5: Save the vector store
        output_dir = self.config["data"]['vector_db_path']
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, db_name)
        vector_store.save_local(save_path)
        self.logger.debug(f"Vector database saved to {save_path}.")
    
    def _initialize_embeddings(self) -> OllamaEmbeddings:
        """
        Initialize the embedding model based on config.
        """
        self.logger.debug("Initializing embeddings...")
        embeddings = OllamaEmbeddings(
            model=self.config["embeddings"]["model"],
            base_url=self.config["embeddings"].get("base_url", "http://localhost:11434")
        )
        self.logger.debug("Embeddings initialized successfully.")
        return embeddings
    
    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Initialize the text splitter based on config.
        """
        self.logger.debug("Initializing text splitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["embeddings"]["chunk_size"],
            chunk_overlap=self.config["embeddings"]["chunk_overlap"]
        )
        self.logger.debug("Text splitter initialized successfully.")
        return text_splitter
    
    def _load_pdfs(self, pdfs_path: str) -> List[str]:
        """
        Find all PDF files in the specified directory.
        """
        self.logger.debug(f"Searching for PDF files in {pdfs_path}...")
        pdfs = []
        for file in os.listdir(pdfs_path):
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(pdfs_path, file))
        self.logger.debug(f"Found {len(pdfs)} PDF files.")
        return pdfs
    
    def _process_documents(self, pdf_paths: List[str]) -> List[Any]:
        """
        Load and split documents from PDF files.
        """
        self.logger.debug(f"Processing {len(pdf_paths)} PDF files...")
        docs = []
        for pdf in pdf_paths:
            loader = PyMuPDFLoader(pdf)
            temp = loader.load()
            docs.extend(temp)
        
        chunks = self.text_splitter.split_documents(docs)
        self.logger.debug(f"Processed {len(docs)} documents into {len(chunks)} chunks.")
        return chunks
    
    def _initialize_vector_store(self) -> FAISS:
        """
        Initialize an empty FAISS vector store.
        """
        self.logger.debug("Initializing FAISS vector store...")
        dummy_embedding = self.embeddings.embed_query("dummy text")
        index = faiss.IndexFlatIP(len(dummy_embedding))
        
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.logger.debug("FAISS vector store initialized successfully.")
        return vector_store

    


def main():
    # Example usage
    processor = DocumentProcessor()
    processor.process_and_store()
    
    # pdfs_root_path = 'clemson_faculty_docs'
    # pdf_names = ['2018-2019', '2019-2020', '2020-2021']
    # config = read_yaml('config.yaml')
    
    # for name in config['data']['input_dir']:
    #     pdfs_path = os.path.join(pdfs_root_path, name)
    #     processor.process_and_store(pdfs_path, name)


if __name__ == '__main__':
    main()