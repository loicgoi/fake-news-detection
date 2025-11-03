import numpy as np
import pandas as pd
from function_chunk.split_chunk import chunk_text

from chroma.chroma_client import ChromaClient
from chroma.chroma_query import query_collection


class ChromaManager:
    def __init__(self, collection_name):
        self.client = ChromaClient()
        self.collection_name = collection_name
        self._embed_function = None
        self._collection = None

    @property
    def embed_function(self):
        if self._embed_function is None:
            self._embed_function = self.client.get_embedding_function()
        return self._embed_function

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        return self._collection

    @staticmethod
    def normalize_L2(vector):
        """
        Normalise un vecteur pour qu'il ait une norme L2 égale à 1.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def add_dataframe_to_collection(
        self,
        df: pd.DataFrame,
        prefix: str,
        step: int = 50,
        overlap: int = 10,
        batch_size: int = 500,
    ):
        """
        Ajoute un DataFrame complet dans la collection Chroma avec chunks + embeddings.
        """

        # Vérifier que la collection est prête
        if not hasattr(self, 'collection') or self.collection is None:
            raise Exception("Collection ChromaDB non initialisée")

        ids, documents, metadatas = [], [], []
        total_rows = len(df)
        rows_handled = 0
        print(f"· {rows_handled}/{total_rows} rows have been handled")
        for row in df.itertuples():
            chunks = chunk_text(
                row.text,
                step=step,
                overlap=overlap,
            )

            for i, chunk in enumerate(chunks):
                ids.append(f"{prefix}_{getattr(row, 'Index')}_{i}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "title": str(row.title),
                        "subject": str(row.subject),
                        "date": str(row.date),
                        "label": str(row.label),
                        "chunk_index": i,
                    }
                )

                if len(documents) >= batch_size:
                    raw_emb = self.embed_function(documents)
                    normalized_emb = [
                        self.normalize_L2(np.array(emb)) for emb in raw_emb
                    ]
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=normalized_emb,
                    )
                    ids, documents, metadatas = [], [], []
            rows_handled += 1
            if rows_handled % batch_size == 0:
                print(f"· {rows_handled}/{total_rows} rows have been handled")
        # Dernier batch
        if len(documents) > 0:
            raw_emb = self.embed_function(documents)
            normalized_emb = [self.normalize_L2(np.array(emb)) for emb in raw_emb]
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=normalized_emb,
            )

    def query(self, text, n_results):
        return query_collection(self.collection, text, n_results)
