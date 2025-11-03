import os
import time
import requests

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
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)

        # Récupération de l'URL Ollama
        ollama_base = os.getenv("OLLAMA_API_BASE", "http://ollama:11434")
        
        # Attendre qu'Ollama soit vraiment prêt
        self._wait_for_ollama(ollama_base, model_name)
        
        # IMPORTANT: ChromaDB ajoute automatiquement /api aux endpoints
        # Il faut donc lui donner l'URL de base SANS /api
        ollama_url = ollama_base

        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name=model_name,
            url=ollama_url
        )

    def _wait_for_ollama(self, base_url: str, model_name: str, max_retries: int = 10):
        """
        Attend qu'Ollama soit prêt et que le modèle soit chargé.
        """
        print(f"Vérification de la disponibilité d'Ollama sur {base_url}...")
        
        for attempt in range(max_retries):
            try:
                # Test 1: Vérifier qu'Ollama répond
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    
                    # Test 2: Vérifier que le modèle est disponible
                    if any(model_name in name for name in model_names):
                        print(f"Ollama prêt avec le modèle {model_name}")
                        
                        # Test 3: Vérifier que l'embedding fonctionne
                        test_payload = {
                            "model": model_name,
                            "input": "test"
                        }
                        embed_response = requests.post(
                            f"{base_url}/api/embed", 
                            json=test_payload, 
                            timeout=10
                        )
                        
                        if embed_response.status_code == 200:
                            print(f"Embeddings fonctionnels")
                            return
                        else:
                            print(f"Embeddings pas encore prêts (status {embed_response.status_code})")
                    else:
                        print(f"Modèle {model_name} pas encore disponible. Modèles: {model_names}")
                else:
                    print(f"Ollama répond avec status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Tentative {attempt + 1}/{max_retries}: Ollama pas encore accessible ({e})")
            
            if attempt < max_retries - 1:
                time.sleep(5)
        
        raise Exception(f"Impossible de se connecter à Ollama après {max_retries} tentatives")

    def get_client(self):
        """
        Retourne le client Chroma
        """
        return self.client

    def get_embedding_function(self):
        """
        Retourne la fonction d'embedding
        """
        if self.embedding_function is None:
            max_retries = 5
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    self.embedding_function = (
                        embedding_functions.OllamaEmbeddingFunction(
                            model_name=self.model_name,
                            url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
                        )
                    )
                    embed_test = self.embedding_function(["test"])
                    print(f"connexion à Ollama réussie -> {embed_test}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"tentative {attempt + 1}/{max_retries} échouée, nouvelle tentative dans {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        raise Exception(
                            f"impossible de se connecter à Ollama après {max_retries} tentatives: {e}"
                        )

        return self.embedding_function

    def get_or_create_collection(self, name: str):
        """
        Récupère ou crée une collection persistente avec embeddings.
        Gère le cas où une collection existe déjà avec une fonction d'embedding différente.
        """
        try:
            # Essayer de récupérer la collection existante
            collection = self.client.get_collection(name=name)
            return collection
        except Exception:
            # Si elle n'existe pas, la créer avec l'embedding function
            return self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function
            )