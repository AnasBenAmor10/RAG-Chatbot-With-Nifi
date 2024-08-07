import streamlit as st
from streamlit_chat import message
import requests
import re

# Configuration de la page
st.set_page_config(page_title="Orthlane AI Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Orthlane AI Assistant</h1>", unsafe_allow_html=True)

API_ENDPOINT = "http://localhost:8501/orthlanechatbot"

# Initialisation des variables d'état de session
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Barre latérale
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation")

# Réinitialiser l'historique des conversations
if clear_button:
    st.session_state['chat_history'] = []

def generate_response(query, chat_history):
    chat_history_string = ' '.join([f'("{item[0]}", "{item[1]}")' for item in chat_history]) if chat_history else ''
    params = {
        "question": query,
        "chat_history": chat_history_string
    }
    try:
        response = requests.post(API_ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return {"answer": "Sorry, something went wrong.", "source": ""}

def parse_response(response):
    if 'source' in response and 'SOURCES' in response['source']:
        answer, source = response['answer'], response['source']
        pattern = r"Notebook__(.*?)__Note__(.*?)__Id__(.*?)\.enex"
        matches = re.findall(pattern, source)
        formatted_sources = [f"Note: {match[1]} (Notebook: {match[0]}), " for match in matches]
        formatted_sources_string = ', '.join(formatted_sources)
        return {"answer": answer, "source": source, "sources_parsed": formatted_sources_string}
    else:
        return {"answer": response['answer'], "source": response['source'], "sources_parsed": ""}

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        chat_history = st.session_state['chat_history']
        response = generate_response(user_input, chat_history)
        chat_history.append((user_input, response))

if st.session_state['chat_history']:
    with response_container:
        for index, (question, answer) in enumerate(st.session_state['chat_history']):
            answer_and_source = parse_response(answer)
            message(question, is_user=True, key=str(index) + '_user')
            message(answer_and_source['answer'], key=str(index))
            if answer_and_source['source']:
                st.markdown("<b>Orthlane Source:</b> " + answer_and_source['sources_parsed'], unsafe_allow_html=True)
