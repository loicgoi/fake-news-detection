import streamlit as st
from prompt.rag_system import RAGSystem

# Config page
st.set_page_config(
    page_title="Fake News Checker",
    page_icon="📰",
    layout="centered",
)

# Init Session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    with st.spinner("Chargement du RAG..."):
        st.session_state.rag_system = RAGSystem()

# Style.css
st.markdown(
    """
    <style>
    /* Fixe la zone d'input en bas */
    .stChatInputContainer {
        position: fixed;
        bottom: 1rem;
        left: 0;
        right: 0;
        z-index: 100;
        background-color: white;
        padding: 1rem 2rem 0.5rem 2rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }

    /* Laisse de la place en bas pour ne pas masquer le contenu */
    .block-container {
        padding-bottom: 8rem;
    }

    /* Permet le scroll si beaucoup de messages */
    .stApp {
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Fake News Checker")
st.divider()
st.markdown(
    """
    Entrez une news ou un article pour vérifier la véracité de l'information.
    """
)

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat
user_input = st.chat_input("Écrivez ici votre news...")

if user_input:
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Réponse assistant
    with st.chat_message("assistant"):
        with st.spinner("Vérification de la véracité de la news..."):
            try:
                result = st.session_state.rag_system.analyze_article(user_input)
                if "Erreur" in result:
                    st.error(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                else:
                    predicted_label, confidence, justification = st.session_state.rag_system.evaluation_rag()
                    final_message = f"""
            **ANALYSIS RESULT**

            **Label :** {predicted_label}  
            **Trust :** `{confidence}%`  

            **Justification :**
            {justification}

            ---
            *Analysis based on comparison with {len(st.session_state.rag_system.search_results["documents"][0])} verified articles from the database.*
                    """
                    st.markdown(final_message, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": final_message})

            except Exception as e:
                error_msg = f"Erreur pendant l'analyse : {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
