import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

message_history = st.session_state['message_history']

# Display existing history
for message in message_history:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here...')
thread_id = '1'
config = {'configurable': {'thread_id': thread_id}}

if user_input:

    # Show user message
    message_history.append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Single Assistant Bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Thinking…"):
            response = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            ai_message = response['messages'][-1].content

        placeholder.text(ai_message)

    # Save assistant message
    message_history.append({'role':'assistant', 'content':ai_message})
