import os
import ollama

class PromptBuilder():

    """
        Prend un texte d'article et retourne une prédiction True/Fake
        en utilisant RAG (Recherche + LLM via Ollama).
    """

    def __init__(self, article_text, model_embedding, model_llm):
        self.article_text = article_text
        self.model_embedding = model_embedding
        self.model_llm = model_llm

        ollama.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    
    def build_context_for_prompt(self, search_results):
        # Rechercher les articles les plus similaires dans la base vectorielle
        # Prend le retour de la fonction query_collection qui a était vectorisé au préalable

        # Récupérer les documents les plus proches
        similar_docs = search_results["documents"][0]
        similar_meta = search_results["metadatas"][0]

        # Construire le contexte à donner au LLM
        context = "\n\n".join([
            f"- Sujet : {meta['subject']}\n  Date : {meta['date']}\n  Label : {meta['label']}\n  Texte : {doc}..."
            for doc, meta in zip(similar_docs, similar_meta)
        ])
        return context
    
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
        # Appele le modèle de langage pour obtenir la classification
        response = ollama.generate(
            model=self.model_llm,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'num_predict': 100,
                'timeout': 120000
            }
        )

        # Retourne la réponse
        return response["response"]