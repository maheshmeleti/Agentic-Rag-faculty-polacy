# frontend/streamlit_app.py
import streamlit as st
import requests
import os
from typing import List
from utils.common_functions import read_yaml

config = read_yaml('config.yaml')
# API configuration
# API_BASE_URL = "http://localhost:8000"  # Update with your FastAPI server URL
# API_BASE_URL = "http://backend:8000"
API_BASE_URL = config['backend']['base_url']


def get_processed_folders() -> List[str]:
    """Fetch available processed folders from the backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/processed_folders")
        if response.status_code == 200:
            return response.json().get("folders", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching folders: {str(e)}")
    return []

def main():
    st.title("RAG Agent Interface")

    # Initialize session state
    if 'processed_folders' not in st.session_state:
        st.session_state.processed_folders = get_processed_folders()
    
    # Document Upload Section
    st.header("Upload Documents")
    folder_name = st.text_input("Enter folder name for documents")
    uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True)
    
    if st.button("Upload"):
        if not folder_name:
            st.error("Please enter a folder name")
            return
        if not uploaded_files:
            st.error("Please select at least one file")
            return
            
        files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
        response = requests.post(
            f"{API_BASE_URL}/upload",
            params={"folder_name": folder_name},
            files=files
        )
        
        if response.status_code == 200:
            st.success("Files uploaded successfully!")
            st.session_state.current_folder = folder_name
        else:
            st.error(f"Error uploading files: {response.json().get('detail', 'Unknown error')}")

    # Document Processing Section
    if 'current_folder' in st.session_state:
        st.header("Process Documents")
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                response = requests.post(
                    f"{API_BASE_URL}/process",
                    params={"folder_name": st.session_state.current_folder}
                )
                
                if response.status_code == 200:
                    st.success("Documents processed successfully!")
                else:
                    st.error(f"Error processing documents: {response.json().get('detail', 'Unknown error')}")

    # Query Section
    st.header("Ask a Question")
    
    # Display available processed folders and allow multi-selection
    if st.session_state.processed_folders:
        selected_folders = st.multiselect(
            "Select processed folders to query from",
            options=st.session_state.processed_folders,
            default=st.session_state.processed_folders[:1]  # Default to first folder
        )
    else:
        st.warning("No processed folders available. Please upload and process documents first.")
        selected_folders = []
    
    query = st.text_area("Enter your question")
    
    if st.button("Submit Query"):
        if not query:
            st.error("Please enter a question")
            return
        if not selected_folders:
            st.error("Please select at least one processed folder")
            return
            
        with st.spinner("Generating response..."):
            try:
                query_data = {
                    "messages": [{"content": query}],
                    "folder_names": selected_folders
                }
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=query_data
                )
                
                if response.status_code == 200:
                    st.subheader("Response")
                    st.write(response.json().get("response", "No response content"))
                else:
                    st.error(f"Error processing query: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

    # Refresh folder list button
    if st.button("Refresh Folder List"):
        st.session_state.processed_folders = get_processed_folders()
        if st.session_state.processed_folders:
            st.success(f"Found {len(st.session_state.processed_folders)} processed folders")
        else:
            st.warning("No processed folders found")

if __name__ == "__main__":
    main()