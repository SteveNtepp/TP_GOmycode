import streamlit as st
import nltk
import pandas as pd
import requests
import io
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import time
import urllib.parse

# --- INITIALISATION NLTK ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# --- CONFIGURATION ---
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS1g6r2EAc7-KgGDkkMTULS_Vl2AFx50QVQjCaHeyuhHOIFLtug4KPaq6aMYiMAQfWh6_dFzr5LO00L/pub?gid=0&single=true&output=csv"
NUMERO_WA = "237679648336"
LIEN_CALENDAR = "https://calendar.app.google/DgFJZkPYehjGzLUD8"
COULEUR_INDIGO = "#4F46E5"


# --- CHARGEMENT DES DONNÉES DEPUIS GOOGLE SHEETS ---
@st.cache_data(ttl=300)
def load_data_from_sheets(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))

        # Normalisation des noms de colonnes pour éviter l'erreur 'catégorie'
        # On remplace les accents et on met en minuscules
        df.columns = [
            c.strip().lower().replace('é', 'e').replace('à', 'a')
            for c in df.columns
        ]

        qa_list = []
        categories = set()
        HIDDEN_CATEGORIES = ["salutations", "aide"]

        # Mapping flexible des noms de colonnes
        col_cat = 'categorie' if 'categorie' in df.columns else df.columns[0]
        col_ques = 'question' if 'question' in df.columns else df.columns[1]
        col_rep = 'reponse' if 'reponse' in df.columns else ('reponse' if 'reponse' in df.columns else df.columns[2])

        for _, row in df.iterrows():
            cat = str(row[col_cat]).strip()
            ques = str(row[col_ques]).strip()
            rep = str(row[col_rep]).strip()

            if cat.lower() != "nan" and ques.lower() != "nan":
                qa_list.append({'categorie': cat, 'question': ques, 'reponse': rep})
                if cat.lower() not in HIDDEN_CATEGORIES:
                    categories.add(cat)

        return qa_list, sorted(list(categories))
    except Exception as e:
        st.error(f"⚠️ Erreur de lecture : Vérifiez que vos colonnes Sheets sont 'Catégorie', 'Question', 'Réponse'.")
        st.info(f"Détail technique : {e}")
        return [], []


qa_data, all_categories = load_data_from_sheets(CSV_URL)


# --- FONCTIONS LOGIQUES ---
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    # Utilisation des stopwords français
    stop_words = set(stopwords.words('french'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]


def get_local_response(query, selected_category):
    query_tokens = preprocess(query)
    max_similarity = -1
    # [cite_start]Réponse basée sur le script de vente en cas d'échec [cite: 25, 57]
    best_response = "Désolé, je n'ai pas la réponse exacte. Mais notre équipe est disponible immédiatement sur WhatsApp pour vous aider !"

    target_cats = [selected_category.lower(), "salutations", "aide"]
    filtered_data = [item for item in qa_data if item['categorie'].lower() in target_cats]

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

    st.markdown(f"""
        <style>
        .stApp {{ background-color: #F8F9FE; }}
        .indigo-text {{ color: {COULEUR_INDIGO} !important; font-weight: 800; margin-bottom: 20px; }}
        .stChatMessage {{ border-radius: 15px; background-color: white; border: 1px solid #E0E4F5; }}

        .stButton>button {{ 
            border-radius: 10px; border: 1px solid {COULEUR_INDIGO}; color: {COULEUR_INDIGO}; 
            background-color: white; font-weight: 600;
        }}
        .stButton>button:hover {{ background-color: {COULEUR_INDIGO}; color: white; }}

        .btn-container {{ display: flex; justify-content: flex-end; margin-top: 25px; margin-right: 25px; }}
        .cta-whatsapp {{
            background-color: #25D366; color: white !important; padding: 12px 24px; border-radius: 50px;
            text-decoration: none !important; font-weight: bold; font-size: 14px; display: flex; align-items: center; gap: 10px;
            box-shadow: 0 4px 15px rgba(37, 211, 102, 0.2); transition: all 0.3s ease;
        }}
        .cta-whatsapp:hover {{ transform: scale(1.05); text-decoration: none !important; }}
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f'<h1 class="indigo-text">Smix Academy</h1>', unsafe_allow_html=True)
        st.link_button("🟢 Parler à un conseiller", f"https://wa.me/{NUMERO_WA}", use_container_width=True)
        st.link_button("📅 Prendre un RDV", LIEN_CALENDAR, use_container_width=True)
        st.divider()
        sujet = st.selectbox("🎯 Quelle thématique vous intéresse ?", options=all_categories, index=None)
        if st.button("🗑️ Réinitialiser le Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown(f'<h1 class="indigo-text">Smix Sales Assistant</h1>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        # [cite_start]Configuration des suggestions dynamiques selon le script de vente [cite: 5, 23, 34]
        suggestions = {
            "Inscription": ["Comment s'inscrire ?", "Dates limites", "Documents requis"],
            "Paiement": ["Tarifs formation", "Modalités de paiement", "Est-ce trop cher ?"],
            "Carrière": ["Débouchés métiers", "Certificat reconnu ?", "Aide au recrutement"],
            "Pédagogie": ["Programme détaillé", "Formation en ligne ?", "Accompagnement pratique"]
        }

        st.write("💡 **Questions fréquentes :**")
        opts = suggestions.get(sujet, ["En savoir plus"])
        cols = st.columns(len(opts))
        for i, opt in enumerate(opts):
            with cols[i]:
                if st.button(opt, use_container_width=True):
                    st.session_state.temp_prompt = opt

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

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

                placeholder = st.empty()
                full_res = ""
                for word in response.split():
                    full_res += word + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.02)
                placeholder.markdown(full_res)

                # [cite_start]Lien WhatsApp avec message pré-rempli [cite: 2, 25]
                msg_wa = f"Bonjour Smix Academy ! Je m'intéresse à la formation {sujet}. J'ai une question : {prompt}"
                msg_encoded = urllib.parse.quote(msg_wa)
                lien_wa_complet = f"https://wa.me/{NUMERO_WA}?text={msg_encoded}"

                st.markdown(f"""
                <div class="btn-container">
                    <a href="{lien_wa_complet}" target="_blank" class="cta-whatsapp">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16">
                          <path d="M13.601 2.326A7.854 7.854 0 0 0 7.994 0C3.627 0 .068 3.558.064 7.926c0 1.399.366 2.76 1.057 3.965L0 16l4.204-1.102a7.933 7.933 0 0 0 3.79.965h.004c4.368 0 7.926-3.558 7.93-7.93A7.898 7.898 0 0 0 13.6 2.326zM7.994 14.521a6.573 6.573 0 0 1-3.356-.92l-.24-.144-2.494.654.666-2.433-.156-.251a6.56 6.56 0 0 1-1.007-3.505c0-3.626 2.957-6.584 6.591-6.584a6.56 6.56 0 0 1 4.66 1.931 6.557 6.557 0 0 1 1.928 4.66c-.004 3.639-2.961 6.592-6.592 6.592zm3.615-4.934c-.197-.099-1.17-.578-1.353-.646-.182-.065-.315-.099-.445.099-.133.197-.513.646-.627.775-.114.133-.232.148-.43.05-.197-.1-.836-.308-1.592-.985-.59-.525-.985-1.175-1.103-1.372-.114-.198-.011-.304.088-.403.087-.088.197-.232.296-.346.1-.114.133-.198.198-.33.065-.134.034-.248-.015-.347-.05-.099-.445-1.076-.612-1.47-.16-.389-.323-.335-.445-.34-.114-.007-.247-.007-.38-.007a.729.729 0 0 0-.529.247c-.182.198-.691.677-.691 1.654 0 .977.71 1.916.81 2.049.098.133 1.394 2.132 3.383 2.992.47.205.84.326 1.129.418.475.152.904.129 1.246.08.38-.058 1.171-.48 1.338-.943.164-.464.164-.86.114-.943-.049-.084-.182-.133-.38-.232z"/>
                        </svg>
                        Finaliser sur WhatsApp
                    </a>
                </div>
                """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Bonjour ! Pour commencer, choisissez une thématique dans la barre latérale.")


if __name__ == "__main__":
    main()