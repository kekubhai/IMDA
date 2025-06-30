from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

class MemoryManager:
    """
    Manages conversational memory and stores/retrieves past fault cases using Chroma vector DB.
    """
    def __init__(self, persist_directory: str = "chroma_db"):
        self.memory = ConversationBufferMemory()
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            )
        except Exception as e:
            print(f"Warning: Could not initialize Chroma vector DB: {e}")
            self.vector_db = None

    def add_case(self, case_text: str):
        """
        Adds a fault case to the vector DB.
        """
        if self.vector_db:
            try:
                # Add unique ID to avoid conflicts
                import uuid
                unique_id = str(uuid.uuid4())
                self.vector_db.add_texts([case_text], ids=[unique_id])
            except Exception as e:
                print(f"Error adding case to vector DB: {e}")

    def retrieve_similar(self, query: str, k: int = 3):
        """
        Retrieves similar past cases from the vector DB.
        """
        if self.vector_db:
            try:
                docs = self.vector_db.similarity_search(query, k=k)
                return [doc.page_content for doc in docs] if docs else []
            except Exception as e:
                print(f"Error retrieving similar cases: {e}")
                return []
        return [] 