import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot, retrieve_all_threads, save_thread_to_db, load_thread_from_db, delete_thread_from_db
import uuid

# *************************** UI CSS ******************************************
st.markdown("""
<style>
/* Sidebar buttons */
button[kind="secondary"], button[kind="primary"] {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

/* Delete icon */
button[data-testid="baseButton-secondary"] {
    padding-left: 4px !important;
    padding-right: 4px !important;
}

/* Active delete icon matches active chat button background */
.stButton > button.active-delete {
    background-color: #4CAF50 !important;
    color: white !important;
}

.stButton > button:hover {
    filter: brightness(0.9);
}
</style>
""", unsafe_allow_html=True)

# *************************** Utility functions ******************************************
def generate_thread_id():
    return str(uuid.uuid4())

def get_previous_chat(thread_id):
    """Load a thread from DB and set session state."""
    st.session_state['thread_id'] = thread_id
    thread_data = load_thread_from_db(thread_id)
    st.session_state['message_history'] = thread_data["messages"]
    st.session_state['chat_titles'][thread_id] = thread_data["title"]

def add_new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_chat_thread(thread_id)
    st.session_state['chat_titles'][thread_id] = "Current Chat"
    st.session_state['message_history'] = []
    save_thread_to_db(thread_id, [], "Current Chat")

def add_chat_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def delete_chat(thread_id):
    # Delete from DB
    delete_thread_from_db(thread_id)

    # Delete from session state
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    if thread_id in st.session_state['chat_titles']:
        del st.session_state['chat_titles'][thread_id]

    # If active thread was deleted, start new empty chat
    if st.session_state['thread_id'] == thread_id:
        new_id = generate_thread_id()
        add_chat_thread(new_id)
        st.session_state['thread_id'] = new_id
        st.session_state['message_history'] = []
        st.session_state['chat_titles'][new_id] = "Current Chat"
        save_thread_to_db(new_id, [], "Current Chat")

# *************************** Session State Initialization ***************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}
    # Load titles for all threads
    for t_id in st.session_state['chat_threads']:
        thread_data = load_thread_from_db(t_id)
        st.session_state['chat_titles'][t_id] = thread_data["title"]

# Ensure current thread exists
if st.session_state['thread_id'] not in st.session_state['chat_titles']:
    st.session_state['chat_titles'][st.session_state['thread_id']] = "Current Chat"

message_history = st.session_state['message_history']
thread_id = st.session_state['thread_id']

# *************************** Sidebar UI *******************************************
st.sidebar.title('Chatbot using LangGraph')
st.sidebar.button('New Chat', on_click=add_new_chat)
st.sidebar.header('My Conversations')

for t_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(t_id, "Untitled Chat")
    is_active = (t_id == st.session_state['thread_id'])
    button_label = f"👉 {title}" if is_active else title

    with st.sidebar.container():
        col1, col2 = st.sidebar.columns([0.82, 0.18])
        with col1:
            st.sidebar.button(
                button_label,
                key=f"thread_btn_{t_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=get_previous_chat,
                args=(t_id,)
            )
        with col2:
            css_class = "active-delete" if is_active else ""
            col2.button(
                "❌",
                key=f"delete_btn_{t_id}",
                help="Delete this chat",
                on_click=delete_chat,
                args=(t_id,),
                use_container_width=True,
            )

# *************************** Display Chat History ***************************
for message in message_history:
    with st.chat_message(message['role']):
        st.text(message['content'])

# *************************** User Input ***************************
user_input = st.chat_input('Type here...')
config = {'configurable': {'thread_id': thread_id}}

if user_input:
    # Append user message to history first
    message_history.append({'role': 'user', 'content': user_input})

    # Auto-generate title if "Current Chat"
    if st.session_state['chat_titles'][thread_id] == "Current Chat":
        short_title = user_input.strip()[:40] + ("…" if len(user_input.strip()) > 40 else "")
        st.session_state['chat_titles'][thread_id] = short_title

    # Save updated thread to DB (messages + title)
    save_thread_to_db(thread_id, message_history, st.session_state['chat_titles'][thread_id])

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
        ai_message = full_response
        message_history.append({'role': 'assistant', 'content': ai_message})

    # Save updated thread to DB again after assistant response
    save_thread_to_db(thread_id, message_history, st.session_state['chat_titles'][thread_id])
