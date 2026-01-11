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
nltk.download('averaged_perceptron_tagger')

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
    best_response = "Désolé, je n'ai pas trouvé de réponse exacte. Pouvez-vous reformuler ?"

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
    st.set_page_config(page_title="SmixBot Pro", page_icon="🤖", layout="centered")

    # --- 2. INJECTION CSS CUSTOM ---
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; }
        .stChatMessage { border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 10px; }
        .stButton>button { border-radius: 20px; width: 100%; }
        </style>
        """, unsafe_allow_html=True)

    # --- 1. BARRE LATÉRALE AVEC EXPANDER ---
    with st.sidebar:
        st.title("🚀 Smix Academy")

        with st.expander("💡 Aide & Utilisation"):
            st.write("""
            1. Sélectionnez un thème ci-dessous.
            2. Posez vos questions sur la formation.
            3. Utilisez les boutons rapides pour gagner du temps.
            """)

        st.divider()
        st.subheader("⚙️ Configuration")
        sujet = st.selectbox(
            "Choisissez votre thématique :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Sélectionner..."
        )

        if st.button("🗑️ Effacer la discussion"):
            st.session_state.messages = []
            st.rerun()

    # --- ZONE DE CHAT ---
    st.title("Chatbot SmixBot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # --- 3. QUICK REPLIES (Boutons de suggestions) ---
        st.write(f"Sujet : **{sujet}**")
        cols = st.columns(2)
        with cols[0]:
            if st.button("📋 Détails du programme"):
                st.session_state.temp_prompt = "Quels sont les détails du programme ?"
        with cols[1]:
            if st.button("⏳ Durée et horaires"):
                st.session_state.temp_prompt = "Quelle est la durée de la formation ?"

        # Affichage de l'historique
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Gestion de la saisie (input direct ou bouton rapide)
        prompt = st.chat_input("Votre message...")
        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # --- 4. INDICATEUR VISUEL D'ÉTAT ---
                with st.status("🔍 SmixBot analyse votre demande...", expanded=False) as status:
                    response = get_response(prompt, sujet)
                    time.sleep(1)  # Simulation recherche
                    status.update(label="✅ Réponse trouvée !", state="complete", expanded=False)

                # Effet Streaming
                placeholder = st.empty()
                full_response = ""
                for word in response.split():
                    full_response += word + " "
                    placeholder.markdown(full_res := full_response + "▌")
                    time.sleep(0.06)
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Bienvenue ! Veuillez sélectionner une thématique dans le menu de gauche pour commencer à discuter.")


if __name__ == "__main__":
    main()