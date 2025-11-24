from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, ChannelVersions
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import uuid
from dotenv import load_dotenv
load_dotenv()

model_gen = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="google/gemma-2-2b-it",
    repo_id="openai/gpt-oss-20b",
    # repo_id="MiniMaxAI/MiniMax-M2",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    # repo_id="moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
generator_llm = ChatHuggingFace(llm=model_gen)

# state
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = generator_llm.invoke(messages)
    return {'messages': [response]}


conn = sqlite3.connect('chatbot.db', check_same_thread=False)

# check point
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# thread_id = '1'
# config = {'configurable': {'thread_id': thread_id}}
# response = chatbot.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

def retrieve_all_threads(): 
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


def delete_thread_from_db(thread_id):
    """
    Delete all checkpoints for a specific thread_id from the SQLite DB.
    """
    conn = checkpointer.conn  # get the sqlite connection used in checkpointer
    cursor = conn.cursor()

    # Delete all rows where thread_id matches
    cursor.execute("""
        DELETE FROM checkpoints
        WHERE thread_id = ?
    """, (thread_id,))

    conn.commit()


def save_thread_to_db(thread_id, messages, title):
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "chat"}}

    # Get existing snapshot (returns dict or None)
    snapshot = checkpointer.get(config)

    if snapshot is None:
        existing_values = {}
        new_versions = {}
    else:
        # snapshot itself is a dict, may contain channel_values & channel_versions
        existing_values = snapshot.get("channel_values", {})
        new_versions = snapshot.get("channel_versions", {})

    # Update values
    existing_values["messages"] = messages
    existing_values["title"] = title

    # Generate a unique checkpoint ID
    checkpoint_id = str(uuid.uuid4())

    checkpoint = {
        "v": 1,
        "ts": None,
        "id": checkpoint_id,
        "channel_values": existing_values,
        "channel_versions": new_versions,
        "pending_sends": {},
        "versions_seen": {},
    }

    # Persist checkpoint
    checkpointer.put(config, checkpoint, metadata={}, new_versions=new_versions)


def load_thread_from_db(thread_id):
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "chat"}}
    snapshot = checkpointer.get(config)

    if snapshot is None:
        return {"messages": [], "title": "Untitled Chat"}

    # Use get() safely because snapshot is a dict
    channel_values = snapshot.get("channel_values", {})
    messages = channel_values.get("messages", [])
    title = channel_values.get("title", "Untitled Chat")

    return {"messages": messages, "title": title}