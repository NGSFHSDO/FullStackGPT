import time
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="☠️",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token 
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


# 파일첨부시에만 load, split, embed, store, retrive함
@st.cache_data(show_spinner="I'm embedding you")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder( 
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings()

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embedder)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state.messages.append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        


def paint_history():
    for message in st.session_state.messages:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


memory = ConversationSummaryBufferMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=80,
    memory_key="chat_history"
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")
st.markdown(
    """
    Who the fuck are you?
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Whatever",
        type=["pdf", "txt", "docx"]
    )

# 파일첨부시 chat_input container가 뜨게, 처음 실행 시 파일이 없거나, 파일을 삭제하면 현재 세션을 빈 리스트로
if file:
    retriever = embed_file(file)
    send_message("Lets Fucking Go", "ai", save=False)
    paint_history()
    message = st.chat_input("LFG")
    if message:
        send_message(message, "user")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(chat_history=load_memory)
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            memory.save_context(
                {"input": message},
                {"output": response.content},
            )
else:
    st.session_state.messages = []
