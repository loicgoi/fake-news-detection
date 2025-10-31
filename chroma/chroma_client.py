import os

import chromadb
from chromadb.utils import embedding_functions

from chroma.singleton import SingletonMeta


class ChromaClient(metaclass=SingletonMeta):
    """
    Singleton qui gère la connexion à ChromaDB et l'embedding Ollama.
    """

    def __init__(
        self, db_path: str = "./chroma_db/", model_name: str = "all-minilm:latest"
    ):
        self.client = chromadb.PersistentClient(path=db_path)

        ollama_url = os.getenv("OLLAMA_API_BASE", "http://ollama:11434")

        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name=model_name,
            url=ollama_url
        )

    def get_client(self):
        """
        Retourne le client Chroma
        """
        return self.client

    def get_embedding_function(self):
        """
        Retourne la fonction d'embedding
        """
        return self.embedding_function

    def get_or_create_collection(self, name: str):
        """
        Récupère ou crée une collection persistente avec embeddings.
        """
        return self.client.get_or_create_collection(
            name=name
        )
