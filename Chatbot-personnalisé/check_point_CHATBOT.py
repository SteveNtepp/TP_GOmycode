import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# --- INITIALISATION ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Chargement des données avec filtrage des catégories
qa_data = []
all_categories = set()

# Liste des catégories à masquer de la sélection utilisateur
HIDDEN_CATEGORIES = ["Salutations", "Aide"]

try:
    with open("question.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                cat, ques, rep = parts
                qa_data.append({'categorie': cat, 'question': ques, 'reponse': rep})
                if cat not in HIDDEN_CATEGORIES:
                    all_categories.add(cat)
except FileNotFoundError:
    st.error(
        "Le fichier 'question.txt' est introuvable. Vérifiez le chemin : /Users/macbook/PyCharmMiscProject/TP_GOmycode/Chatbot-personnalisé/question.txt")


# 2. Fonction de Prétraitement
def preprocess(sentence):
    words = word_tokenize(sentence.lower())
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

    # On cherche dans : la catégorie sélectionnée ET les catégories masquées (Salutations/Aide)
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

    # Barre latérale pour le choix du sujet (exclut Salutations et Aide)
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

    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'interface de chat
    if sujet:
        st.info(f"📍 Vous discutez de : **{sujet}**")

        # Afficher l'historique
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Saisie utilisateur
        if prompt := st.chat_input("Posez votre question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Réponse du bot (inclut la logique pour les salutations même si non sélectionnées)
            response = get_response(prompt, sujet)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning("👈 Veuillez sélectionner une catégorie dans le menu à gauche pour commencer.")
        st.write(
            "Une fois une catégorie choisie, vous pourrez poser vos questions et SmixBot vous répondra en utilisant les données de formation.")


if __name__ == "__main__":
    main()