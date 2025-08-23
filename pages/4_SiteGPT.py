import streamlit as st
import asyncio
import nest_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from urllib.parse import urlparse
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda



st.set_page_config(
page_title="Site",
page_icon="📃",
)

st.title("SiteGPT")
st.markdown(
"""
Welcome!
"""
)

with st.sidebar:
    url = st.text_input("url", placeholder="https://example.com")


### 나무위키처럼 자바스크립트로 떡칠된 사이트가 있으면 쓰자. headless라 느릴수도...
# # Apply nest_asyncio to allow asyncio.run() within an existing event loop
# nest_asyncio.apply()
# # Set the event loop policy for Windows
# if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# html2TextTransformer = Html2TextTransformer()

# async def load_documents(url):
#     loader = AsyncChromiumLoader([url])
#     documents = loader.load()
#     return documents

# if url:
#     documents = asyncio.run(load_documents(url))
#     st.write(documents)
#     transformed = html2TextTransformer.transform_documents(documents)
#     st.write(documents)
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
)


answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
        
    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
    """
)

def get_answer(input):
    docs = input["docs"]
    question = input["question"]
    answer_chain = answer_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answer_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }



choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful).

            Cite sources and return the sources of the answers as they are, do not change them.


            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    docs = inputs["answers"]
    question = inputs["question"]
    condensed = "\n\n".join(
        f"Answer: {doc['answer']}\nSource: {doc['source']}\n" for doc in docs
    )
    choose_chain = choose_prompt | llm
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )
        


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose() # <header>이하 싹다 없애기
    if footer:
        footer.decompose()
    return (
        str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")
    )


@st.cache_data(show_spinner="Scrapping...")
def load_url(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        # filter_urls=["https://platform.openai.com/docs/pricing"],
        # 대괄호안 url만 통과시키겠다.
        # filter_urls=[r"^(.*\/ranking\/).*"] # /rangking/을 포함하는 url만 통과시키겠다. 반대는 r"^(?!.*\/ranking\/).*"
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1 # 1초에 request 1번
    documents = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    # # URL을 파일 경로로 사용하기 위해 안전한 이름으로 변경
    # parsed_url = urlparse(url)
    # host = parsed_url.netloc.replace("www.", "").replace(".", "_")
    # cache_dir = LocalFileStore(f"./.cache/site_embeddings/{host}")
    # cache_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Who the fuck are you?")
    else: 
        retriever = load_url(url)
        query = st.text_input("Ask bout this site")
        if query:
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answer) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.write(result.content.replace("$", "\$"))