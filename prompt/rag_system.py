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
                # 1. Similarité sémantique
                label_matches = 0
                total_similarity = 0
                
                for label in true_labels:
                    if label.lower() == predicted_label.lower():
                        label_matches += 1
                        total_similarity += 1.0
                    elif (label.lower() == "true" and predicted_label.lower() == "fake") or \
                        (label.lower() == "fake" and predicted_label.lower() == "true"):
                        total_similarity += 0.0  # Opposition totale
                    else:
                        total_similarity += 0.5  # Incertain ou autres labels
                
                # 2. Base confidence avec pondération
                base_confidence = (total_similarity / len(true_labels)) * 100
                
                # 3. Cohérence LLM avec scores variables
                llm_confidence_indicators = {
                    "True": {
                        "strong": ["verified", "credible", "official", "factual", "reliable", "trusted"],
                        "medium": ["consistent", "plausible", "reasonable", "logical"],
                        "weak": ["likely", "possible", "probable"]
                    },
                    "Fake": {
                        "strong": ["sensational", "conspiracy", "fabricated", "misleading", "deceptive"],
                        "medium": ["exaggerated", "unverified", "questionable", "dubious"], 
                        "weak": ["possibly false", "potentially misleading", "unconfirmed"]
                    }
                }
                
                # 4. Score de cohérence nuancé
                justification_lower = justification.lower()
                coherence_score = 0
                
                indicators = llm_confidence_indicators.get(predicted_label, {})
                for strength, words in indicators.items():
                    for word in words:
                        if word in justification_lower:
                            if strength == "strong":
                                coherence_score += 15
                            elif strength == "medium":
                                coherence_score += 8
                            else:  # weak
                                coherence_score += 3
                            break  # Un mot trouvé par force suffit
                
                # 5. Facteur de certitude basé sur la longueur et détail de la justification
                justification_confidence = min(20, len(justification.split()) / 5)  # +1% par 5 mots
                
                # 6. Diversité des sources (pénalité si peu de sources différentes)
                unique_sources = len(set(meta.get('subject', '') for meta in self.search_results["metadatas"][0]))
                diversity_bonus = min(10, unique_sources * 2)  # +2% par source unique
                
                # 7. Calcul final avec pondérations
                confidence = min(100, 
                    base_confidence * 0.6 +           # 60% base similarity
                    coherence_score * 0.3 +           # 30% cohérence LLM  
                    justification_confidence * 0.05 + # 5% qualité justification
                    diversity_bonus * 0.05            # 5% diversité sources
                )
                
                # 8. Ajustement final basé sur le nombre de résultats
                results_count = len(true_labels)
                if results_count < 3:
                    confidence *= 0.8  # Réduction si peu de résultats
                elif results_count > 15:
                    confidence = min(100, confidence * 1.1)  # Léger boost si beaucoup de résultats
                    
            else:
                # Confidence par défaut avec variation aléatoire
                import random
                confidence = 45 + random.randint(0, 10)  # Entre 45% et 55%
        else:
            import random
            confidence = 40 + random.randint(0, 20)  # Entre 40% et 60% si pas de résultats

        return predicted_label, round(confidence, 2), justification
