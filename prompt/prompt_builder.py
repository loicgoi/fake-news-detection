import os
import ollama


class PromptBuilder:
    """
    Prend un texte d'article et retourne une prédiction True/Fake
    en utilisant RAG (Recherche + LLM via Ollama).
    """

    def __init__(self, article_text, model_embedding, model_llm):
        self.article_text = article_text
        self.model_embedding = model_embedding
        self.model_llm = model_llm

        ollama_host = os.getenv("OLLAMA_API_BASE", "http://ollama:11434")
        ollama.api_base = ollama_host
    
    def build_context_for_prompt(self, search_results):
        # Rechercher les articles les plus similaires dans la base vectorielle
        # Prend le retour de la fonction query_collection qui a était vectorisé au préalable

        # Récupérer les documents les plus proches
        similar_docs = search_results["documents"][0]
        similar_meta = search_results["metadatas"][0]

        # Construire le contexte à donner au LLM
        context_parts = []
        for doc, meta in zip(similar_docs, similar_meta):
            # Gérer les documents vides ou None
            doc_preview = str(doc)[:200] + "..." if doc else "[Document vide]"
            context_parts.append(
                f"- Sujet : {meta.get('subject', 'N/A')}\n"
                f"  Date : {meta.get('date', 'N/A')}\n"
                f"  Label : {meta.get('label', 'N/A')}\n"
                f"  Texte : {doc_preview}"
            )
        
        return "\n\n".join(context_parts) if context_parts else "Aucun contexte disponible."
    
    def build_prompt(self, context):
        # Construire le prompt complet pour le LLM
        prompt = f"""
                You are an expert in detecting fake news. 
                You have a knowledge base containing articles that have already been verified, with their metadata:
                - subject: main topic
                - date: publication date
                - label: “True” or “Fake”
                - text: content of the article.

                Here are some similar articles from your database:
                {context}

                Your task is to analyze the following new article and determine whether it is “True” or “Fake.”

                New article to analyze:
        ---
        {self.article_text}
        ---

        Respond only with:
        Label: "True" or "Fake"
        Justification: in 2 sentences maximum, based on the similarities or tone of the article.
        """
        return prompt

    def predict_label(self, prompt):
        try:
            import time
            time.sleep(5)
            response = ollama.generate(
                model=self.model_llm,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 100,
                    'timeout': 300000
                }
            )
            if "response" in response:
                return response["response"]
            else:
                return f"Erreur : clé 'response' manquante dans la réponse Ollama : {response}"
        except Exception as e:
            return f"Erreur pendant l'appel Ollama : {e}"