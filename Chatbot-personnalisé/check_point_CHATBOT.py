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

# --- CONFIGURATION ---
LIEN_WA = "https://wa.me/237679648336"
LIEN_CALENDAR = "https://calendar.app.google/DgFJZkPYehjGzLUD8"
COULEUR_INDIGO = "#4F46E5"

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


# --- FONCTIONS LOGIQUES ---
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


def get_local_response(query, selected_category):
    query_tokens = preprocess(query)
    max_similarity = -1
    best_response = "Désolé, je n'ai pas la réponse exacte. Mais notre équipe est disponible immédiatement sur WhatsApp pour vous aider !"

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

    # CSS : Indigo Blue et Bouton CTA Stylisé (Sans surlignage et avec marge augmentée)
    st.markdown(f"""
        <style>
        .stApp {{ background-color: #F8F9FE; }}

        /* Titres Indigo */
        .indigo-text {{ 
            color: {COULEUR_INDIGO} !important; 
            font-weight: 800; 
            margin-bottom: 20px;
        }}

        .stChatMessage {{ border-radius: 15px; background-color: white; border: 1px solid #E0E4F5; }}

        /* Boutons d'action rapides */
        .stButton>button {{ 
            border-radius: 10px; 
            border: 1px solid {COULEUR_INDIGO}; 
            color: {COULEUR_INDIGO}; 
            background-color: white;
            font-weight: 600;
        }}
        .stButton>button:hover {{ 
            background-color: {COULEUR_INDIGO}; 
            color: white; 
        }}

        /* Bouton CTA WhatsApp Amélioré */
        .btn-container {{
            display: flex;
            justify-content: flex-end;
            margin-top: 25px; /* Marge augmentée pour aérer le bloc */
            margin-bottom: 10px;
        }}
        .cta-whatsapp {{
            background-color: #25D366;
            color: white !important;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none !important; /* Suppression du surlignage */
            font-weight: bold;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(37, 211, 102, 0.2);
            transition: all 0.3s ease;
        }}
        .cta-whatsapp:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(37, 211, 102, 0.4);
            text-decoration: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f'<h1 class="indigo-text">Smix Academy</h1>', unsafe_allow_html=True)
        st.link_button("🟢 Parler à un conseiller", LIEN_WA, use_container_width=True)
        st.link_button("📅 Prendre un RDV", LIEN_CALENDAR, use_container_width=True)
        st.divider()
        sujet = st.selectbox("🎯 Quelle thématique vous intéresse ?", options=sorted(list(all_categories)), index=None)
        if st.button("🗑️ Réinitialiser le Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown(f'<h1 class="indigo-text">Smix Sales Assistant</h1>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # 4 Boutons d'action par catégorie
        suggestions = {
            "Inscription": ["Comment s'inscrire ?", "Documents requis", "Dates limites", "Conditions d'admission"],
            "Carrière": ["Débouchés métiers", "Aide au recrutement", "Stages", "Partenariats entreprises"],
            "Paiement": ["Tarifs formation", "Modalités de paiement", "Bourses disponibles", "Remboursement"],
            "Pédagogie": ["Programme détaillé", "Supports de cours", "Examens", "Projets pratiques"]
        }

        st.write("💡 **Actions recommandées :**")
        opts = suggestions.get(sujet, [])
        cols = st.columns(2)
        for i, opt in enumerate(opts):
            with cols[i % 2]:
                if st.button(opt, use_container_width=True):
                    st.session_state.temp_prompt = opt

        # Affichage historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        prompt = st.chat_input("Posez votre question à l'assistant...")
        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = get_local_response(prompt, sujet)

                # Effet de dactylographie
                placeholder = st.empty()
                full_res = ""
                for word in response.split():
                    full_res += word + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.03)
                placeholder.markdown(full_res)

                # --- LE CTA WHATSAPP (Design Premium) ---
                st.markdown(f"""
                <div class="btn-container">
                    <a href="{LIEN_WA}" target="_blank" class="cta-whatsapp">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16">
                          <path d="M13.601 2.326A7.854 7.854 0 0 0 7.994 0C3.627 0 .068 3.558.064 7.926c0 1.399.366 2.76 1.057 3.965L0 16l4.204-1.102a7.933 7.933 0 0 0 3.79.965h.004c4.368 0 7.926-3.558 7.93-7.93A7.898 7.898 0 0 0 13.6 2.326zM7.994 14.521a6.573 6.573 0 0 1-3.356-.92l-.24-.144-2.494.654.666-2.433-.156-.251a6.56 6.56 0 0 1-1.007-3.505c0-3.626 2.957-6.584 6.591-6.584a6.56 6.56 0 0 1 4.66 1.931 6.557 6.557 0 0 1 1.928 4.66c-.004 3.639-2.961 6.592-6.592 6.592zm3.615-4.934c-.197-.099-1.17-.578-1.353-.646-.182-.065-.315-.099-.445.099-.133.197-.513.646-.627.775-.114.133-.232.148-.43.05-.197-.1-.836-.308-1.592-.985-.59-.525-.985-1.175-1.103-1.372-.114-.198-.011-.304.088-.403.087-.088.197-.232.296-.346.1-.114.133-.198.198-.33.065-.134.034-.248-.015-.347-.05-.099-.445-1.076-.612-1.47-.16-.389-.323-.335-.445-.34-.114-.007-.247-.007-.38-.007a.729.729 0 0 0-.529.247c-.182.198-.691.677-.691 1.654 0 .977.71 1.916.81 2.049.098.133 1.394 2.132 3.383 2.992.47.205.84.326 1.129.418.475.152.904.129 1.246.08.38-.058 1.171-.48 1.338-.943.164-.464.164-.86.114-.943-.049-.084-.182-.133-.38-.232z"/>
                        </svg>
                        Finaliser sur WhatsApp
                    </a>
                </div>
                """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Bonjour ! Sélectionnez une thématique dans la barre latérale pour activer l'assistant de vente.")


if __name__ == "__main__":
    main()