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

# --- CONFIGURATION WHATSAPP ---
NUMERO_WA = "23767964336"
LIEN_WA = f"https://wa.me/{NUMERO_WA}?text=Bonjour%20Smix%20Academy,%20j'aimerais%20en%20savoir%20plus%20sur%20vos%20formations."

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
    best_response = "Désolé, je n'ai pas trouvé de réponse précise. Pouvez-vous reformuler ou contacter un conseiller ?"

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

    # --- CSS PERSONNALISÉ ---
    st.markdown("""
        <style>
        .stChatMessage { border-radius: 15px; padding: 15px; border: 1px solid rgba(128,128,128,0.2); margin-bottom: 15px; background-color: rgba(128,128,128,0.05); }
        .stButton>button { border-radius: 8px; border: 1px solid #4CAF50; color: #4CAF50; background-color: transparent; font-size: 0.85rem; }
        .stButton>button:hover { background-color: #4CAF50; color: white; }
        /* Style spécial pour le bouton WhatsApp */
        div[data-testid="stMarkdownContainer"] > p > a > button { background-color: #25D366 !important; color: white !important; border: none !important; }
        </style>
        """, unsafe_allow_html=True)

    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.title("🚀 Smix Academy")

        # Bouton WhatsApp permanent dans la sidebar
        st.link_button("🟢 Parler à un conseiller", LIEN_WA, use_container_width=True)

        st.divider()
        sujet = st.selectbox(
            "🎓 Thématique de formation :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Sélectionnez un sujet"
        )
        if st.button("🗑️ Réinitialiser le Chat"):
            st.session_state.messages = []
            st.rerun()

    st.title("SmixBot Assistant 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        st.markdown(f"💡 *Suggestions pour **{sujet}*** :")

        # Suggestions dynamiques
        suggestions = []
        if sujet == "Inscription":
            suggestions = ["Comment s'inscrire ?", "Documents requis", "Dates limites"]
        elif sujet == "Carrière":
            suggestions = ["Débouchés métiers", "Aide au recrutement", "Stages"]
        elif sujet == "Paiement":
            suggestions = ["Tarifs formation", "Modalités de paiement", "Bourses disponibles"]
        elif sujet == "Pédagogie":
            suggestions = ["Programme détaillé", "Supports de cours", "Projets pratiques"]

        cols = st.columns(len(suggestions))
        for i, option in enumerate(suggestions):
            with cols[i]:
                if st.button(option):
                    st.session_state.temp_prompt = option

        # Affichage historique
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Saisie
        prompt = st.chat_input("Posez votre question ici...")

        if "temp_prompt" in st.session_state:
            prompt = st.session_state.temp_prompt
            del st.session_state.temp_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.status("🔍 Recherche en cours...", expanded=False) as status:
                    response = get_response(prompt, sujet)
                    time.sleep(0.6)
                    status.update(label="✅ Réponse trouvée", state="complete")

                placeholder = st.empty()
                full_res = ""
                for word in response.split():
                    full_res += word + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.04)
                placeholder.markdown(full_res)

                # Petit rappel WhatsApp après les réponses d'aide ou de paiement
                if sujet in ["Paiement", "Inscription"]:
                    st.caption("Besoin d'une assistance immédiate ?")
                    st.link_button("💬 Finaliser sur WhatsApp", LIEN_WA)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Bonjour ! Sélectionnez une **thématique** à gauche pour commencer.")
        st.link_button("📞 Nous contacter directement", LIEN_WA)


if __name__ == "__main__":
    main()