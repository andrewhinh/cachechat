# Importing the libraries
import os
import math
import requests

import bs4
from dotenv import load_dotenv
import nltk
import numpy as np
import openai
import streamlit as st
from streamlit_chat import message
import textract
import tiktoken
import uuid
import validators


# Helper variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-3.5-turbo"
embed_context_window, embed_model = (
    8191,
    "text-embedding-ada-002",
)  # https://platform.openai.com/docs/guides/embeddings/second-generation-models

nltk.download(
    "punkt"
)  # Load the cl100k_base tokenizer which is designed to work with the ada-002 model (engine)
tokenizer = tiktoken.get_encoding("cl100k_base")
download_chunk_size = 128  # TODO: Find optimal chunk size for downloading files
split_chunk_tokens = 300  # TODO: Find optimal chunk size for splitting text


# Helper functions
def get_num_tokens(text):  # Count the number of tokens in a string
    return len(
        tokenizer.encode(text, disallowed_special=())
    )  # disallowed_special=() removes the special tokens


#   TODO:
#   Currently, any sentence that is longer than the max number of tokens will be its own chunk
#   This is not ideal, since this doesn't ensure that the chunks are of a maximum size
#   Find a way to split the sentence into chunks of a maximum size
def split_into_many(text):  # Split text into chunks of a maximum number of tokens
    sentences = nltk.tokenize.sent_tokenize(text)  # Split the text into sentences
    total_tokens = [
        get_num_tokens(sentence) for sentence in sentences
    ]  # Get the number of tokens for each sentence

    chunks = []
    tokens_so_far = 0
    chunk = []
    for sentence, num_tokens in zip(sentences, total_tokens):
        if not tokens_so_far:  # If this is the first sentence in the chunk
            if (
                num_tokens > split_chunk_tokens
            ):  # If the sentence is longer than the max number of tokens, add it as its own chunk
                chunk.append(sentence)
                chunks.append(" ".join(chunk))
                chunk = []
        else:  # If this is not the first sentence in the chunk
            if (
                tokens_so_far + num_tokens > split_chunk_tokens
            ):  # If the sentence would make the chunk longer than the max number of tokens, add the chunk to the list of chunks
                chunks.append(" ".join(chunk))
                chunk = []
                tokens_so_far = 0

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += num_tokens + 1

    # In case the file is smaller than the max number of tokens, add the last chunk
    if not chunks:
        chunks.append(" ".join(chunk))
    return chunks


def embed(prompt):  # Embed the prompt using the ada-002 model
    embeds = []
    if (
        type(prompt) == str
    ):  # Split prompt (str of n token length) into chunks if n > context_window
        if get_num_tokens(prompt) > embed_context_window:
            prompt = split_into_many(prompt)
        else:
            embeds = openai.Embedding.create(input=prompt, model=embed_model)["data"]
    if (
        not embeds
    ):  # Split prompt (list of n chunks of max size split_chunk_tokens) into multiple lists if n * split_chunk_tokens > context_window
        max_num_chunks = (
            embed_context_window // split_chunk_tokens
        )  # Number of chunks that can fit in the context window
        for i in range(0, math.ceil(len(prompt) / max_num_chunks)):
            embeds.extend(
                openai.Embedding.create(
                    input=prompt[i * max_num_chunks : (i + 1) * max_num_chunks],
                    model=embed_model,
                )["data"]
            )
    return embeds


def embed_files(filenames):  # Create embeddings for files
    sources = []
    chunks = []
    vectors = []

    for filename in filenames:
        extracted_text = (
            textract.process(filename)
            .decode("utf-8")  # Extracted text is in bytes, convert to string
            .encode("ascii", "ignore")  # Remove non-ascii characters
            .decode()  # Convert back to string
        )
        sources.append(filename)
        chunks.append(split_into_many(extracted_text))
        vectors.append([x["embedding"] for x in embed(chunks[-1])])

    return sources, chunks, vectors


def embed_url(url):  # Create embeddings for a url
    source = None
    chunks = []
    vectors = []
    filename = None

    if validators.url(url, public=True):  # Verify url is a valid and public
        response = requests.get(url)
        header = response.headers["Content-Type"]
        is_application = (
            header.split("/")[0] == "application"
        )  # Check if the url is a file

        if (
            is_application
        ):  # If url is a file, download it and extract text later as a file
            filetype = header.split("/")[1]
            url_parts = url.split("/")
            filename = str(
                "./"
                + "|".join(
                    url_parts[:-1] + [url_parts[-1].split(".")[0]]
                )  # Replace / with | in the filename to avoid issues with the file path and remove the file extension since it may not match the actual filetype
                + "."
                + filetype
            )
            with requests.get(url, stream=True) as stream_response:  # Download the file
                stream_response.raise_for_status()
                with open(filename, "wb") as file:
                    for chunk in stream_response.iter_content(
                        chunk_size=download_chunk_size
                    ):
                        file.write(chunk)
        else:  # If url is a webpage, use BeautifulSoup to extract the text
            soup = bs4.BeautifulSoup(response.text)
            extracted_text = (
                soup.get_text()  # Extract the text from the webpage
                .encode("ascii", "ignore")  # Remove non-ascii characters
                .decode()  # Convert back to string
            )
            source = url
            chunks.append(split_into_many(extracted_text))
            vectors.append([x["embedding"] for x in embed(chunks[-1])])

    if filename:
        file_sources, file_chunks, file_vectors = embed_files([filename])
        source = file_sources
        chunks = file_chunks
        vectors = file_vectors

    return source, chunks, vectors


def ask(questions, answers, sources, vectors, chunks):  # Ask a question
    messages = [
        {"role": "user", "content": questions[0]}
    ]  # Add the first question to the list of messages
    message(
        questions[0],
        is_user=True,
        avatar_style="thumbs",
    )  # Display user's first question

    if len(questions) > 1 and answers:  # If this is not the first question
        for interaction, message in enumerate(
            [message for pair in zip(answers, questions[1:]) for message in pair]
        ):  # Add and display the messages from the previous conversation in the correct order
            if (
                interaction % 2 == 0
            ):  # If the message is an answer, add and display it as a chatbot message
                messages.append({"role": "assistant", "content": message})
                message(
                    message,
                    avatar_style="fun-emoji",
                )
            else:  # If the message is a question, add and display it as a user message
                messages.append({"role": "user", "content": message})
                message(
                    message,
                    is_user=True,
                    avatar_style="thumbs",
                )

    answer = openai.ChatCompletion.create(model=llm_model, messages=messages)[
        "choices"
    ][0]["message"][
        "content"
    ]  # Get the answer from the chatbot
    return answer


# Main function
def main():
    st.title("CacheChat")  # Creating the chatbot interface

    sources, chunks, vectors = [], [], []  # Sources, chunks, and vectors of embeddings
    uploaded_files = st.file_uploader(
        "Choose file(s):", accept_multiple_files=True
    )  # Upload (a) file(s)
    if uploaded_files:  # If (a) file(s) is/are uploaded, create embeddings
        with st.spinner("Processing..."):
            file_sources, file_chunks, file_vectors = embed_files(
                [uploaded_file.name for uploaded_file in uploaded_files]
            )
            sources.extend(file_sources)
            chunks.extend(file_chunks)
            vectors.extend(file_vectors)
    with st.form(clear_on_submit=True):
        uploaded_url = st.text_input(
            "Enter a URL:", placeholder="https://www.africau.edu/images/default/sample.pdf"
        )  # Enter a URL
        upload_url_button = st.button("Add URL")  # Add URL button
        if upload_url_button:  # If a URL is entered, create embeddings
            if uploaded_url:  # If URL is not empty
                with st.spinner("Processing..."):
                    url_source, url_chunks, url_vectors = embed_url(list(uploaded_url))
                    sources.append(url_source)
                    chunks.extend(url_chunks)
                    vectors.extend(url_vectors)

    st.divider()  # Create a divider between the uploads and the chat

    questions = []  # User's questions
    answers = []  # Chatbot's answers

    input_container = st.container()  # container for text box
    response_container = st.container()  # container for chat history

    with input_container:
        with st.form(clear_on_submit=True):
            uploaded_question = st.text_input(
                "Enter your input:",
                placeholder="e.g: Summarize the research paper in 3 sentences.",
                key="input",
            )  # question text box
            uploaded_question_button = st.form_submit_button(
                label="Send"
            )  # send button

        if (
            uploaded_question_button and uploaded_question
        ):  # if send button is pressed and text box is not empty
            with response_container:
                questions.append(uploaded_question)
                answer = ask(questions, answers, sources, chunks, vectors)
                answers.append(answer)
                message(
                    answers[-1],
                    avatar_style="fun-emoji",
                )


if __name__ == "__main__":
    main()
