import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import time

# --- INITIALISATION NLTK ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# --- CONFIGURATION DES CHEMINS ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "question.txt")

# 1. Chargement des données
qa_data = []
all_categories = set()
HIDDEN_CATEGORIES = ["Salutations", "Aide"]

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                cat, ques, rep = parts
                qa_data.append({'categorie': cat, 'question': ques, 'reponse': rep})
                if cat not in HIDDEN_CATEGORIES:
                    all_categories.add(cat)
except FileNotFoundError:
    st.error("Fichier de données introuvable.")


# 2. Prétraitement
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


# 3. Logique de réponse
def get_response(query, selected_category):
    query_tokens = preprocess(query)
    if not query_tokens:
        return "Je vous écoute, n'hésitez pas à poser une question précise."

    max_similarity = -1
    best_response = "Désolé, je n'ai pas trouvé de réponse précise. Essayez de reformuler ou changez de thématique."

    target_categories = [selected_category] + HIDDEN_CATEGORIES
    filtered_data = [item for item in qa_data if item['categorie'] in target_categories]

    for item in filtered_data:
        item_tokens = preprocess(item['question'])
        union = set(query_tokens).union(item_tokens)
        if not union: continue
        similarity = len(set(query_tokens).intersection(item_tokens)) / float(len(union))
        if similarity > max_similarity and similarity > 0.1:
            max_similarity = similarity
            best_response = item['reponse']
    return best_response


# 4. INTERFACE PRINCIPALE
def main():
    # Configuration de la page
    st.set_page_config(page_title="SmixBot", page_icon="🤖", layout="centered")

    # --- INJECTION CSS PERSONNALISÉ ---
    st.markdown(f"""
        <style>
        /* Couleurs du thème */
        .stApp {{
            background-color: #f5f5f5; /* Gris très clair */
            font-family: 'Montserrat', sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: #d342ca; /* Magenta/Rose */
            color: white;
        }}
        .stButton>button {{
            background-color: #6420ff; /* Violet primaire */
            color: white;
            border-radius: 10px;
            border: none;
        }}
        /* Style des bulles de chat */
        .stChatMessage {{
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        </style>
        """, unsafe_allow_html=True)

    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.title("🚀 Smix Academy")

        with st.expander("💡 Aide & Utilisation"):
            st.write("""
            1. Sélectionnez un sujet de formation.
            2. Posez vos questions dans le chat.
            3. Utilisez les suggestions pour aller plus vite.
            """)

        st.divider()
        st.header("Configuration")
        sujet = st.selectbox(
            "Sujet de la formation :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Choisir un thème..."
        )

        if st.button("🔄 Nouvelle discussion"):
            st.session_state.messages = []
            st.rerun()

    # --- ZONE DE CHAT ---
    st.title("SmixBot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # Organisation visuelle avec colonnes
        col_info, col_stat = st.columns([3, 1])
        with col_info:
            st.info(f"📍 Sujet : **{sujet}**")
        with col_stat:
            st.metric("Messages", len(st.session_state.messages))

        # Affichage de l'historique
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Quick Replies (Suggestions)
        st.write("---")
        st.caption("Suggestions :")
        q1, q2 = st.columns(2)
        suggestion = None
        with q1:
            if st.button(f"Infos sur {sujet}"):
                suggestion = f"Peux-tu me donner des infos sur {sujet} ?"
        with q2:
            if st.button("Comment s'inscrire ?"):
                suggestion = "Comment s'inscrire à cette formation ?"

        # Saisie utilisateur (manuelle ou via suggestion)
        prompt = st.chat_input("Posez votre question ici...")
        if suggestion:
            prompt = suggestion

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Indicateur visuel d'état
            with st.chat_message("assistant"):
                with st.status("SmixBot analyse votre demande...", expanded=False) as status:
                    response = get_response(prompt, sujet)
                    time.sleep(1)  # Simulation de recherche
                    status.update(label="Réponse trouvée !", state="complete", expanded=False)

                # Effet Streaming
                placeholder = st.empty()
                full_response = ""
                for word in response.split():
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.06)
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Page d'accueil si aucun sujet n'est choisi
        st.warning("👈 Veuillez sélectionner un sujet dans la barre latérale pour commencer l'aventure.")
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=200)


if __name__ == "__main__":
    main()