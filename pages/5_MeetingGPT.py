import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(
    temperature=0.1
)

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov", "webm"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
        transcript_path = video_path.rsplit(".", 1)[0] + ".txt"
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Chunking audio...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, chat_tab = st.tabs(
        ["Transcript", "Summary", "Chat"]
    )

    with transcript_tab:
        with open(transcript_path, "r", encoding="utf-8") as f:
            st.write(f.read())

    with summary_tab:
        LFG = st.button("LFG")
        if LFG:
            loader = TextLoader(transcript_path, encoding="utf-8")
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = loader.load_and_split(text_splitter=splitter)


            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke(
                {"text": docs[0].page_content},
            )


            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final, consolidated summary.
                An existing summary is provided: {existing_summary}
                Refine the existing summary by incorporating the following new context.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary. If the context isn't useful, return the original summary.
                Output ONLY the refined summary, without any other text.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Summarizing chunk {i+1}/{len(docs)-1}...")    
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
            st.write(summary)
    with chat_tab:
        st.write("귀찮아서 안할꺼임.")

        
