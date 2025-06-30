from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

class MemoryManager:
    """
    Manages conversational memory and stores/retrieves past fault cases using Chroma vector DB.
    """
    def __init__(self, persist_directory: str = "chroma_db"):
        self.memory = ConversationBufferMemory()
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=SentenceTransformerEmbeddings()
        )

    def add_case(self, case_text: str):
        """
        Adds a fault case to the vector DB.
        """
        self.vector_db.add_texts([case_text])

    def retrieve_similar(self, query: str, k: int = 3):
        """
        Retrieves similar past cases from the vector DB.
        """
        return self.vector_db.similarity_search(query, k=k) 