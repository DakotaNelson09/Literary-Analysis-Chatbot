from openai import OpenAI
import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

user = OpenAI(base_url="https://api.deepseek.com", api_key=st.secrets["deepseek-key"])

#Initialize
if "model" not in st.session_state:
    st.session_state["model"] = "deepseek-chat"
if "chats" not in st.session_state:
    st.session_state.chats = {0: {"title": "New Chat", "messages": []}}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = 0
if "editing_user_msg" not in st.session_state:
    st.session_state.editing_user_msg = None
if "editing_title" not in st.session_state:
    st.session_state.editing_title = None
if "pending_regen" not in st.session_state:
    st.session_state.pending_regen = False

#Load vector store at startup
if "vectorstore" not in st.session_state:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    has_resources = os.path.exists("resources") and any(
        f.endswith(".txt") for f in os.listdir("resources")
    )
    if os.path.exists("knowledge_index"):
        st.session_state.vectorstore = FAISS.load_local(
            "knowledge_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    elif has_resources:
        with st.spinner("Building knowledge base, please wait..."):
            loader = DirectoryLoader(
                "resources",
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=450
            )
            chunks = splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local("knowledge_index")
            st.session_state.vectorstore = vectorstore
    else:
        st.session_state.vectorstore = None

def get_relevant_context(query: str, k: int = 8) -> str:
    """Retrieve the k most relevant chunks from the knowledge base."""
    if st.session_state.vectorstore is None:
        return ""
    docs = st.session_state.vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)

def display_user_message(content: str):
    """Render user message preserving newlines."""
    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.html(f'<p style="white-space: pre-wrap; margin: 0;">{escaped}</p>')

def generate_chat_title(first_message: str) -> str:
    response = user.chat.completions.create(
        model=st.session_state["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate very short chat titles (3-5 words, English only). "
                    "Respond with only the title, no punctuation, no quotes."
                ),
            },
            {"role": "user", "content": f"Title for a chat that starts with: {first_message}"},
        ],
    )
    return response.choices[0].message.content.strip()

def stream_response(messages, query: str):
    """Stream a response using relevant context from the knowledge base."""
    context = get_relevant_context(query)
    tot_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always respond in English. Use the provided context to inform your answers, but you may also draw on your general knowledge when the context does not contain sufficient information."
        }
    ]
    if context:
        tot_messages.append({
            "role": "system",
            "content": f"Use the following relevant information to help answer the user's question:\n\n{context}"
        })
    with st.chat_message("assistant"):
        space = st.empty()
        big_token_cost = ""
        stream = user.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                *tot_messages,
                *[{"role": m["role"], "content": m["content"]} for m in messages],
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            big_token_cost += token
            space.markdown(big_token_cost + "▌")
        space.markdown(big_token_cost)
    return big_token_cost

#Sidebar
with st.sidebar:
    st.title("Slant")
    if st.button("+ New Chat"):
        new_id = max(st.session_state.chats.keys()) + 1
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = new_id
        st.session_state.editing_user_msg = None
        st.session_state.editing_title = None
        st.session_state.pending_regen = False

    st.divider()

    for chat_id in sorted(st.session_state.chats.keys()):
        title = st.session_state.chats[chat_id]["title"]

        if st.session_state.editing_title == chat_id:
            new_title = st.text_input(
                "Rename",
                value=title,
                key=f"title_input_{chat_id}",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", key=f"title_save_{chat_id}"):
                    st.session_state.chats[chat_id]["title"] = new_title
                    st.session_state.editing_title = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"title_cancel_{chat_id}"):
                    st.session_state.editing_title = None
                    st.rerun()
        else:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                if st.button(title, key=f"chat_{chat_id}"):
                    st.session_state.current_chat = chat_id
                    st.session_state.editing_user_msg = None
            with col2:
                if st.button("✏️", key=f"title_edit_{chat_id}"):
                    st.session_state.editing_title = chat_id
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    remaining = sorted(st.session_state.chats.keys())
                    if remaining:
                        st.session_state.current_chat = remaining[0]
                    else:
                        st.session_state.chats[0] = {"title": "New Chat", "messages": []}
                        st.session_state.current_chat = 0
                    st.session_state.editing_user_msg = None
                    st.session_state.editing_title = None
                    st.rerun()

#Active chat
current = st.session_state.chats[st.session_state.current_chat]
messages = current["messages"]

#Render messages
for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            if st.session_state.editing_user_msg == (st.session_state.current_chat, i):
                edited_user = st.text_area(
                    "Edit your message",
                    value=message["content"],
                    key=f"edit_user_{st.session_state.current_chat}_{i}",
                    label_visibility="collapsed",
                )
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("Save", key=f"save_user_{i}"):
                        messages[i]["content"] = edited_user
                        del messages[i + 1:]
                        st.session_state.editing_user_msg = None
                        st.session_state.pending_regen = True
                        st.session_state.pending_query = edited_user
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_user_{i}"):
                        st.session_state.editing_user_msg = None
                        st.rerun()
            else:
                display_user_message(message["content"])
                if st.button("✏️", key=f"edit_btn_{st.session_state.current_chat}_{i}"):
                    st.session_state.editing_user_msg = (st.session_state.current_chat, i)
                    st.rerun()

#Handle pending regeneration after loop
if st.session_state.pending_regen:
    st.session_state.pending_regen = False
    query = st.session_state.get("pending_query", messages[-1]["content"])
    response = stream_response(messages, query)
    messages.append({"role": "assistant", "content": response})
    st.rerun()

#Input
if prompt := st.chat_input("What is up?"):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        display_user_message(prompt)

    if len(messages) == 1:
        current["title"] = generate_chat_title(prompt)

    response = stream_response(messages, prompt)
    messages.append({"role": "assistant", "content": response})
    st.rerun()