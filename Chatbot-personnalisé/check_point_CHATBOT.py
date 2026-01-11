import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import time
import google.generativeai as genai

# --- CONFIGURATION GEMINI ---
# Ta clé est intégrée ici, mais assure-toi qu'elle est active sur Google AI Studio
GEMINI_API_KEY = "AIzaSyAdKQw-DubGqOk-Zr6ST_xQ1UFwFVVxGJc"
genai.configure(api_key=GEMINI_API_KEY)
model_ai = genai.GenerativeModel('gemini-1.5-flash')

# --- INITIALISATION NLTK ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# --- CONFIGURATION DES CHEMINS & LIENS ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "question.txt")
LIEN_WA = "https://wa.me/237679648336"
LIEN_CALENDAR = "https://calendar.app.google/DgFJZkPYehjGzLUD8"

# 1. Chargement des données locales (Base de connaissances officielle)
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
    st.error("Erreur : Le fichier question.txt est introuvable.")


# 2. Prétraitement du texte
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


# 3. Logique de réponse Hybride (Local + Gemini)
def get_hybrid_response(query, selected_category):
    query_tokens = preprocess(query)
    max_similarity = -1
    best_local_response = None

    # A. Recherche dans le fichier local
    target_categories = [selected_category] + HIDDEN_CATEGORIES
    filtered_data = [item for item in qa_data if item['categorie'] in target_categories]

    for item in filtered_data:
        item_tokens = preprocess(item['question'])
        union = set(query_tokens).union(item_tokens)
        if not union: continue
        similarity = len(set(query_tokens).intersection(item_tokens)) / float(len(union))
        if similarity > max_similarity:
            max_similarity = similarity
            best_local_response = item['reponse']

    # B. Décision de réponse
    # Si similarité forte (>0.6), on donne la réponse officielle du fichier texte
    if max_similarity > 0.6:
        return best_local_response

    # Sinon, on utilise Gemini pour répondre intelligemment
    else:
        # Instruction spécifique pour que Gemini agisse en tant qu'expert Smix Academy
        prompt_system = f"""
        Tu es 'Smix Sales Assistant', l'expert de la Smix Academy.
        Contexte actuel : La formation sur la thématique '{selected_category}'.

        Instructions :
        1. Sois professionnel, enthousiaste et concis.
        2. Si la question porte sur des détails techniques (comme le design graphique, l'IA, etc.), explique clairement les bénéfices.
        3. Si tu ne connais pas un prix ou une date spécifique, invite l'utilisateur à cliquer sur le bouton WhatsApp.
        4. Ne mentionne jamais que tu es une IA ou un modèle de langage.

        Question du client : """

        try:
            response = model_ai.generate_content(prompt_system + query)
            if response and response.text:
                return response.text
            else:
                return "C'est une excellente question. Pour vous donner une réponse précise et adaptée à votre projet, je vous suggère d'en discuter directement avec un de nos conseillers sur WhatsApp."
        except Exception:
            return "Je n'ai pas pu générer de réponse pour le moment. Pourriez-vous reformuler ou contacter notre équipe sur WhatsApp ?"


# 4. INTERFACE PRINCIPALE
def main():
    st.set_page_config(page_title="Smix Sales Assistant", page_icon="🤖", layout="centered")

    # --- STYLE GRAPHIQUE INDIGO & CLAIR ---
    st.markdown("""
        <style>
        .stApp { background-color: #F8F9FE; color: #1E1E2F; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E4F5; }
        .stChatMessage { border-radius: 12px; border: 1px solid #E0E4F5; background-color: #FFFFFF; padding: 15px; margin-bottom: 10px; }
        .stButton>button { border-radius: 8px; border: 1px solid #4F46E5; color: #4F46E5; background-color: #FFFFFF; font-weight: 500; }
        .stButton>button:hover { background-color: #4F46E5; color: white; }
        </style>
        """, unsafe_allow_html=True)

    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.title("Smix Academy")
        st.link_button("Parler à un conseiller", LIEN_WA, use_container_width=True)
        st.link_button("Prendre un RDV", LIEN_CALENDAR, use_container_width=True)
        st.divider()

        sujet = st.selectbox(
            "Thématique de formation :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Sélectionnez un sujet"
        )
        if st.button("Réinitialiser le Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- ZONE DE CHAT ---
    st.title("Smix Sales Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # Suggestions interactives
        suggestions = {
            "Inscription": ["Comment s'inscrire ?", "Documents requis"],
            "Carrière": ["Débouchés métiers", "Stages"],
            "Paiement": ["Tarifs formation", "Modalités de paiement"],
            "Pédagogie": ["Programme détaillé", "Projets pratiques"]
        }

        current_suggestions = suggestions.get(sujet, ["Plus d'infos"])

        st.write("**Suggestions rapides :**")
        cols = st.columns(len(current_suggestions) + 1)
        for i, opt in enumerate(current_suggestions):
            with cols[i]:
                if st.button(opt): st.session_state.temp_prompt = opt
        with cols[-1]:
            st.link_button("Prendre RDV", LIEN_CALENDAR)

        # Affichage de l'historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Gestion de l'input utilisateur
        prompt = st.chat_input("Posez votre question ici...")
        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.status("Recherche de la meilleure réponse...", expanded=False) as status:
                    # Ici, on appelle notre logique hybride améliorée
                    response = get_hybrid_response(prompt, sujet)
                    time.sleep(0.5)
                    status.update(label="Analyse terminée", state="complete")

                placeholder = st.empty()
                full_res = ""
                # Effet d'écriture progressive
                for word in response.split():
                    full_res += word + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.04)
                placeholder.markdown(full_res)

                # Relance WhatsApp pour les sujets de conversion
                if sujet in ["Paiement", "Inscription"]:
                    st.write("---")
                    c1, c2 = st.columns(2)
                    c1.link_button("💬 WhatsApp", LIEN_WA, use_container_width=True)
                    c2.link_button("🗓️ Calendrier", LIEN_CALENDAR, use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Veuillez sélectionner une thématique dans la barre latérale pour commencer l'assistance.")


if __name__ == "__main__":
    main()