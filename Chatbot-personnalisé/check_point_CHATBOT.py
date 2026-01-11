import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import time
from PIL import Image

# --- INITIALISATION NLTK ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# --- CONFIGURATION DES CHEMINS ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "question.txt")
logo_path = os.path.join(base_path, "Asset", "logo Smix-transparent.png")

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
    st.error("Fichier 'question.txt' introuvable.")


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
    best_response = "Désolé, je n'ai pas trouvé de réponse précise. Essayez de reformuler."

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
    st.set_page_config(page_title="SmixBot", page_icon="🤖")

    # --- BARRE LATÉRALE ---
    with st.sidebar:
        # Affichage du Logo REDIMENSIONNÉ
        try:
            image = Image.open(logo_path)
            # Nous avons retiré use_container_width et fixé la largeur à 150
            st.image(image, width=150)
        except Exception:
            st.title("🚀 Smix Academy")

        st.divider()
        st.header("Configuration")
        sujet = st.selectbox(
            "Sujet de la formation :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Choisir un thème..."
        )

        if st.button("Nouvelle discussion"):
            st.session_state.messages = []
            st.rerun()

    # --- ZONE DE CHAT ---
    st.title("SmixBot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sujet:
        st.info(f"📍 Discussion sur : **{sujet}**")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Posez votre question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = get_response(prompt, sujet)
                placeholder = st.empty()
                full_response = ""

                # Effet Streaming (Mot par mot)
                for word in response.split():
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.08)

                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("👈 Veuillez sélectionner une catégorie pour commencer.")


if __name__ == "__main__":
    main()