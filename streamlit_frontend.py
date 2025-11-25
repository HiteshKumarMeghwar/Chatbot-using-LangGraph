import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot, retrieve_all_threads, delete_thread_from_db
import uuid


# ----------  CSS  ----------
st.markdown(
    """
    <style>
    /* force same height for both buttons in the same row */
    [data-testid="stSidebar"] button {
        height: 38px !important;
        line-height: 38px !important;
        padding: 0 8px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    /* active chat colour */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# *************************** Utility functions ******************************************
def generate_thread_id():
    return str(uuid.uuid4())

def add_new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_chat_thread(thread_id)
    st.session_state['message_history'] = []

def add_chat_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def delete_chat(thread_id):
    # Delete from DB
    delete_thread_from_db(thread_id)

    # Delete from session state
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)

    # If active thread was deleted, start new empty chat
    if st.session_state['thread_id'] == thread_id:
        new_id = generate_thread_id()
        add_chat_thread(new_id)
        st.session_state['thread_id'] = new_id
        st.session_state['message_history'] = []

def load_conversation(thread_id):
    try:
        messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values
    except Exception:
        return []

    if not messages or "messages" not in messages:
        return []
    else:
        tem_messages = []
        for message in messages['messages']:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            tem_messages.append({'role': role, 'content': message.content})

        return tem_messages

def get_previous_chat(thread_id):
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = load_conversation(thread_id)

def get_preview(messages):
    if not messages:
        return "Empty chat"

    last_msg = messages[-1]["content"].strip()

    if len(last_msg) > 12:
        return last_msg[:12] + "..."
    return last_msg


# *************************** Session State Initialization ***************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
    add_new_chat()


# *************************** Sidebar UI *******************************************
st.sidebar.title('MeghX-Bot')
st.sidebar.button('New Chat', on_click=add_new_chat)
st.sidebar.header('My Conversations')

for t_id in st.session_state['chat_threads'][::-1]:

    messages = load_conversation(t_id)
    prev = get_preview(messages)
    active = (t_id == st.session_state['thread_id'])
    
    # ----------  ONE ROW = two columns inside a container  ----------
    c1, c2 = st.sidebar.columns([0.82, 0.18])
    with c1:
        st.button(
            prev,
            key=f"open_{t_id}",
            use_container_width=True,
            on_click=get_previous_chat,
            args=(t_id,),
            type="primary" if active else "secondary"
        )
    with c2:
        st.button(
            "X",
            key=f"del_{t_id}",
            help="Delete this chat",
            on_click=delete_chat,
            args=(t_id,),
            use_container_width=True
        )



# *************************** Display Chat History ***************************
message_history = st.session_state['message_history']
for message in message_history:
    with st.chat_message(message['role']):
        st.text(message['content'])



# *************************** User Input ***************************
user_input = st.chat_input('MeghX Bot User - Type here...')
thread_id = st.session_state['thread_id']
config = {'configurable': {'thread_id': thread_id}}

if user_input:
    # Append user message to history first and display it
    message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Assistant response
    with st.chat_message('assistant'):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            full_response = ""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                full_response += message_chunk.content
                placeholder.text(full_response)
            message_history.append({'role': 'assistant', 'content': full_response})