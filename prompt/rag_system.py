import re
from chroma.chroma_manager import ChromaManager

from .prompt_builder import PromptBuilder


class RAGSystem:
    def __init__(self, collection_name="news", model_lm="llama3.2:1b"):
        self.chroma_manager = ChromaManager(collection_name)
        self.embedding_functions = self.chroma_manager.embed_function

        self.model_embedding = "all-minilm:latest"
        self.model_llm = model_lm

    def analyze_article(self, article_text):
        """
        Analyse un article et prédit son label (True/Fake) à l'aide du modèle RAG.
        Nettoie également la réponse du LLM pour éviter les doublons.
        """
        try:
            # Recherche des articles similaires
            self.search_results = self.chroma_manager.query(article_text, n_results=10)

            # Construction du prompt
            prompt_builder = PromptBuilder(
                article_text=article_text,
                model_embedding=self.model_embedding,
                model_llm=self.model_llm,
            )

            # Construction du contexte
            context = prompt_builder.build_context_for_prompt(self.search_results)

            # Construction du prompt final
            prompt = prompt_builder.build_prompt(context)

            # Prédiction
            self.response = prompt_builder.predict_label(prompt)

            return self.response

        except Exception as e:
            return f"Erreur lors de l'analyse: {e}"

    def evaluation_rag(self):
        """
        Évaluation améliorée qui utilise mieux les labels des chunks
        """
        if not hasattr(self, 'response'):
            raise Exception("× `self.response` est introuvable")

        # Extraction du label
        llm_text = self.response
        predicted_label = "Incertain"
        
        patterns = [
            r"Label\s*:\s*[\"']?([Tt]rue|[Ff]ake|[Ff]alse)[\"']?",
            r"^[\"']?([Tt]rue|[Ff]ake|[Ff]alse)[\"']?\s*$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, llm_text, re.IGNORECASE | re.MULTILINE)
            if match:
                raw_label = match.group(1).strip('"\'')
                if raw_label.lower() in ["true"]:
                    predicted_label = "True"
                    break
                elif raw_label.lower() in ["fake", "false"]:
                    predicted_label = "Fake" 
                    break

        # Extraction de la justification
        justification = "Analyse basée sur la comparaison avec la base de données."
        justification_patterns = [
            r"Justification\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"Reasoning\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            r"Identify which criteria match[^:]*:\s*(.+)",
        ]
        
        for pattern in justification_patterns:
            match = re.search(pattern, llm_text, re.IGNORECASE | re.DOTALL)
            if match:
                justification = match.group(1).strip()
                justification = re.sub(r'\s+', ' ', justification)
                break

        # Calcul de confiance
        if hasattr(self, 'search_results') and self.search_results:
            true_labels = [meta['label'] for meta in self.search_results["metadatas"][0]]
            
            if true_labels:
                # Compter les matches avec le label prédit
                matches = sum(1 for label in true_labels if label.lower() == predicted_label.lower())
                base_confidence = (matches / len(true_labels)) * 100
                
                # Ajustement basé sur la cohérence du LLM
                llm_confidence_indicators = {
                    "True": ["official", "verified", "factual", "credible"],
                    "Fake": ["sensational", "conspiracy", "unverified", "extraordinary"]
                }
                
                # Vérifier si la justification du LLM est cohérente
                justification_lower = justification.lower()
                coherence_bonus = 0
                
                for indicator in llm_confidence_indicators.get(predicted_label, []):
                    if indicator in justification_lower:
                        coherence_bonus += 10
                
                confidence = min(100, base_confidence + coherence_bonus)
            else:
                confidence = 50.0
        else:
            confidence = 50.0

        return predicted_label, round(confidence, 2), justification
