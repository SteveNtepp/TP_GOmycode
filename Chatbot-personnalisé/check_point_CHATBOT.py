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

# --- CONFIGURATION DES LIENS ---
LIEN_WA = "https://wa.me/237679648336"
LIEN_CALENDAR = "https://calendar.app.google/DgFJZkPYehjGzLUD8"

# --- CHARGEMENT DES DONNÉES ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "question.txt")

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
    st.error("Fichier question.txt introuvable.")


# --- PRÉTRAITEMENT ---
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


# --- LOGIQUE DE RÉPONSE (RECHERCHE LOCALE UNIQUEMENT) ---
def get_local_response(query, selected_category):
    query_tokens = preprocess(query)
    max_similarity = -1
    best_response = "Désolé, je n'ai pas la réponse exacte. Contactez-nous sur WhatsApp pour une aide personnalisée !"

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


# --- INTERFACE ---
def main():
    st.set_page_config(page_title="Smix Sales Assistant", page_icon="🤖", layout="centered")

    # CSS : Indigo Blue pour les titres et boutons
    st.markdown("""
        <style>
        .stApp { background-color: #F8F9FE; }
        /* Indigo Blue pour les titres spécifiques */
        .indigo-title { color: #4F46E5; font-weight: bold; }
        .stChatMessage { border-radius: 12px; background-color: white; border: 1px solid #E0E4F5; }
        .stButton>button { 
            border-radius: 8px; 
            border: 1px solid #4F46E5; 
            color: #4F46E5; 
            background-color: white;
            font-size: 0.8rem;
        }
        .stButton>button:hover { background-color: #4F46E5; color: white; }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        # Titre en Bleu Indigo
        st.markdown('<h1 class="indigo-title">Smix Academy</h1>', unsafe_allow_html=True)
        st.link_button("Parler à un conseiller", LIEN_WA, use_container_width=True)
        st.link_button("Prendre un RDV", LIEN_CALENDAR, use_container_width=True)
        st.divider()
        sujet = st.selectbox("Sélectionnez une thématique :", options=sorted(list(all_categories)), index=None)
        if st.button("Effacer la discussion"):
            st.session_state.messages = []
            st.rerun()

    # Titre principal en Bleu Indigo
    st.markdown('<h1 class="indigo-title">Smix Sales Assistant</h1>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # --- 4 BOUTONS D'ACTION PAR CATÉGORIE ---
        suggestions = {
            "Inscription": ["Comment s'inscrire ?", "Documents requis", "Dates limites", "Conditions d'admission"],
            "Carrière": ["Débouchés métiers", "Aide au recrutement", "Stages", "Partenariats entreprises"],
            "Paiement": ["Tarifs formation", "Modalités de paiement", "Bourses disponibles", "Remboursement"],
            "Pédagogie": ["Programme détaillé", "Supports de cours", "Examens", "Projets pratiques"]
        }

        st.write("📌 **Actions rapides :**")
        opts = suggestions.get(sujet, [])
        cols = st.columns(2)  # Disposition en 2x2 pour la lisibilité
        for i, opt in enumerate(opts):
            with cols[i % 2]:
                if st.button(opt, use_container_width=True):
                    st.session_state.temp_prompt = opt

        # Affichage historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        # Input
        prompt = st.chat_input("Posez votre question...")
        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = get_local_response(prompt, sujet)
                # Effet d'écriture
                placeholder = st.empty()
                full_res = ""
                for word in response.split():
                    full_res += word + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.03)
                placeholder.markdown(full_res)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Bienvenue ! Veuillez choisir une thématique dans la barre latérale pour commencer.")


if __name__ == "__main__":
    main()