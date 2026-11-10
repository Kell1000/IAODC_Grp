import streamlit as st
from rag import build_index, answer_question

st.set_page_config(
    page_title="Ziyad Food",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Ziyad Food")
st.caption("Your AI restaurant assistant")

@st.cache_resource
def load_rag():
    return build_index()

try:
    chunks, embeddings = load_rag()
    st.success("Restaurant knowledge base loaded successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask me about our restaurant...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(
                    question, chunks, embeddings
                )
                st.write(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

except Exception as error:
    st.error(f"Could not load the knowledge base: {error}")
    st.info("Check that Ollama is running and your documents are in the docs folder.")