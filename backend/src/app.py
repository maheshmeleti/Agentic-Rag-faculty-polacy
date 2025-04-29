from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
# from src.agent.react_agent import ReactAgent
from src.agent.self_evaluation_agent import SelfEvaluationAgent
from src.logger import get_logger
from utils.common_functions import read_yaml
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

app = FastAPI(title="RAG Backend API", description="Backend for Retrieval-Augmented Generation (RAG) system.")
logger = get_logger("RAGBackend")

os.makedirs("data/raw/clemson_faculty_docs", exist_ok=True)
os.makedirs("data/processed/clemson_faculty_docs", exist_ok=True)

config = read_yaml('config.yaml')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    messages: List[dict]
    folder_names: List[str]

from src.data_processing import DocumentProcessor

document_processor = DocumentProcessor()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_documents(folder_name: str, files: List[UploadFile] = File(...)):
    logger.info('upload called....')
    try:
        # Create folder if it doesn't exist
        logger.info(f'folder name - {folder_name}')
        file_names = [file.filename for file in files]
        logger.info(f'files - {file_names}')
        folder_path = os.path.join(config['data']['input_dir'], folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save uploaded files
        for file in files:
            file_path = os.path.join(folder_path, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
        
        return {"status": "success", "message": f"Files saved to {folder_name}", "folder": folder_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_documents(folder_name: str):
    try:
        document_processor = DocumentProcessor()
        document_processor.process_and_store(folder_name)
        return {"status": "success", "message": f"Documents processed for {folder_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_agent(request: QueryRequest):
    logger.info(f'query request called')
    logger.info(f'messages - {request.messages[0]["content"]}')
    logger.info(f'folders - {request.folder_names}')
    try:
        agent = SelfEvaluationAgent(folders=request.folder_names)
        # agent = ReactAgent(folders=request.folder_names)
        graph = agent.build_agent()

        query = {"messages": [HumanMessage(request.messages[0]["content"])]}
        # Modify your agent to accept multiple folders
        response = graph.invoke(query)

        logger.info('Response created')
        response_messages = []
        for output in graph.stream(query):
            for key, value in output.items():
                if key == "generate" or key == 'generate_quiz':
                    for msg in value["messages"]:
                        response_messages.append(msg)

        if response_messages:
            response = response_messages[-1]
        else:
            response = "I didn't get a response. Please try again."
        
        logger.info(f'response message - {response}')

        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# backend/app.py
@app.get("/processed_folders")
async def list_processed_folders():
    logger.info('processed_folders called')
    try:
        # List folders in your processed directory
        processed_dir = config['data']['vector_db_path']  # Or your specific processed folder path
        folders = [f for f in os.listdir(processed_dir) 
                  if os.path.isdir(os.path.join(processed_dir, f))]
        logger.info(processed_dir)
        return {"status": "success", "folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)