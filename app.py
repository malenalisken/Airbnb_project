"""Streamlit chat UI for guests and hosts."""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from agent import new_chat

st.set_page_config(page_title='Airbnb-prisagent Oslo', page_icon='🏠')
st.title('🏠 Airbnb-prisagent for Oslo')
st.caption('Prisestimater fra gruppens egen Gradient Boosting-modell, forklart med Gemini.')
role = st.radio('Jeg er', ['Gjest', 'Utleier'], horizontal=True)
if st.session_state.get('role') != role:
    st.session_state.role = role
    st.session_state.messages = []
    st.session_state.pop('gemini_chat', None)

st.info('Modellen bruker boligtype, gjester, soverom, bad, netter, tid til innsjekking og geografiske koordinater. Valuta må verifiseres mot datakilden.')
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('Beskriv boligen eller spør om en pris...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        try:
            if 'gemini_chat' not in st.session_state:
                st.session_state.gemini_chat = new_chat(role)
            response = st.session_state.gemini_chat.send_message(prompt)
            answer = response.text or 'Jeg fikk ikke et tekstsvar. Prøv å formulere spørsmålet på nytt.'
            st.write(answer)
            st.session_state.messages.append({'role': 'assistant', 'content': answer})
        except Exception as exc:
            st.error(f'Kunne ikke hente svar: {exc}')
