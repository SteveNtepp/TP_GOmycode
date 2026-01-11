import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

# --- INITIALISATION ---
# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('punkt_tab')  # CRUCIAL : Résout l'erreur LookupError sur Streamlit Cloud
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 1. Chargement des données avec gestion du chemin universel
qa_data = []
all_categories = set()
HIDDEN_CATEGORIES = ["Salutations", "Aide"]

# Détection automatique du dossier du script pour trouver question.txt
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "question.txt")

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
    st.error("Erreur : Le fichier 'question.txt' est introuvable dans le dépôt GitHub.")

# 2. Fonction de Prétraitement
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    # Utilisation de stopwords en anglais comme dans votre script original
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]

# 3. Logique de recherche (Hybride : Catégorie choisie + Salutations/Aide)
def get_response(query, selected_category):
    query_tokens = preprocess(query)
    if not query_tokens:
        return "Je vous écoute, n'hésitez pas à poser une question précise."

    max_similarity = -1
    best_response = "Désolé, je n'ai pas trouvé de réponse précise. Essayez de reformuler ou changez de catégorie."

    # Recherche étendue à la catégorie choisie + messages de base (salutations/aide)
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

# 4. INTERFACE STREAMLIT
def main():
    st.set_page_config(page_title="SmixBot", page_icon="🤖")
    st.title("SmixBot 🤖")

    with st.sidebar:
        st.header("Configuration")
        st.write("Choisissez un thème pour orienter la discussion.")

        sujet = st.selectbox(
            "Sujet de la formation :",
            options=sorted(list(all_categories)),
            index=None,
            placeholder="Choisir un thème..."
        )

        if st.button("Nouvelle discussion"):
            st.session_state.messages = []
            st.rerun()

    # Initialisation de l'historique dans le session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage du chat
    if sujet:
        st.info(f"📍 Sujet actuel : **{sujet}**")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Posez votre question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = get_response(prompt, sujet)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning("👈 Veuillez sélectionner une catégorie dans la barre latérale.")
        st.write("SmixBot répondra à vos questions sur les formations une fois le sujet choisi.")

if __name__ == "__main__":
    main()