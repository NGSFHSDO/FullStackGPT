import streamlit as st
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler, get_openai_callback
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser



class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)
    
output_parser = JsonOutputParser()



st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)


### Stuff라 이부분이 필요없을거같긴한데.. MapReduce를 위해 남겨두자...
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a helpful assistant that is role playing as a teacher.
                        
                    Based ONLY on the following context make 5 questions to test the user's knowledge about the text.
                    
                    Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    Context: {context}
                """,
            )
        ]
    )

question_chain = {"context": format_docs} | question_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making Quiz...")
### retriever의 리턴값 document가 content, metadata등이 포함된 구조라 해시가 불가능 -> docs가 바뀌어도 실행안됨
def final_chain(_docs, topic):
    chain = question_chain | (lambda x: x.additional_kwargs["function_call"]["arguments"]) | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wiki(topic):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(topic)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search in Wikipedia")
        if topic:
            docs = search_wiki(topic)



if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = final_chain(docs, topic if topic else file.name)
    with st.form(key="question_form"):
        for a in response["questions"]:
            st.write(a["question"])
            ## 빈리스트 안만들고 append까지
            value = st.radio(
                "Select an option", 
                [answer["answer"] for answer in a["answers"]],
                index=None, ## -> 기본값으로 아무것도 선택안되게
            )
            if ({"answer": value, "correct": True} in a["answers"]):
                st.success("Correct!")
            elif value is not None: 
                st.error("Incorrect!")    
        button = st.form_submit_button("Submit")
