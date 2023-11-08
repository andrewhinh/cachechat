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
from streamlit_chat import message as show_message
import textract
import tiktoken
import uuid
import validators


# Helper variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Load OpenAI API key from .env file

llm_model = "gpt-4-1106-preview"  # https://platform.openai.com/docs/guides/chat/introduction
llm_context_window = (
    128000  # https://platform.openai.com/docs/guides/chat/managing-tokens
)
embed_context_window, embed_model = (
    8191,
    "text-embedding-ada-002",
)  # https://platform.openai.com/docs/guides/embeddings/second-generation-models
nltk.download(
    "punkt"
)  # Download the nltk punkt tokenizer for splitting text into sentences
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # Load the cl100k_base tokenizer which is designed to work with the ada-002 model (engine)

download_chunk_size = 128  # TODO: Find optimal chunk size for downloading files
split_chunk_tokens = 300  # TODO: Find optimal chunk size for splitting text
num_citations = 5  # TODO: Find optimal number of citations to give context to the LLM

# Streamlit settings
user_avatar_style = "fun-emoji"  # https://www.dicebear.com/styles
assistant_avatar_style = "bottts-neutral"


# Helper functions
def get_num_tokens(text):  # Count the number of tokens in a string
    return len(
        tokenizer.encode(text, disallowed_special=())
    )  # disallowed_special=() removes the special tokens)


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


def embed(prompt):  # Embed the prompt
    embeds = []
    if type(prompt) == str:
        if (
            get_num_tokens(prompt) > embed_context_window
        ):  # If token_length of prompt > context_window
            prompt = split_into_many(prompt)  # Split prompt into multiple chunks
        else:  # If token_length of prompt <= context_window
            embeds = openai.Embedding.create(input=prompt, model=embed_model)[
                "data"
            ]  # Embed prompt
    if not embeds:  # If the prompt was split into/is set of chunks
        max_num_chunks = (
            embed_context_window // split_chunk_tokens
        )  # Number of chunks that can fit in the context window
        for i in range(
            0, math.ceil(len(prompt) / max_num_chunks)
        ):  # For each batch of chunks
            embeds.extend(
                openai.Embedding.create(
                    input=prompt[i * max_num_chunks : (i + 1) * max_num_chunks],
                    model=embed_model,
                )["data"]
            )  # Embed the batch of chunks
    return embeds  # Return the list of embeddings


def embed_file(filename):  # Create embeddings for a file
    source_type = "file"  # To help distinguish between local/URL files and URLs
    file_source = ""  # Source of the file
    file_chunks = []  # List of file chunks (from the file)
    file_vectors = []  # List of lists of file embeddings (from each chunk)

    try:
        extracted_text = (
            textract.process(filename)
            .decode("utf-8")  # Extracted text is in bytes, convert to string
            .encode("ascii", "ignore")  # Remove non-ascii characters
            .decode()  # Convert back to string
        )
        if not extracted_text:  # If the file is empty
            raise Exception
        os.remove(
            filename
        )  # Remove the file from the server since it is no longer needed
        file_source = filename
        file_chunks = split_into_many(extracted_text)  # Split the text into chunks
        file_vectors = [x["embedding"] for x in embed(file_chunks)]  # Embed the chunks
    except Exception:  # If the file cannot be extracted, return empty values
        if os.path.exists(filename):  # If the file still exists
            os.remove(
                filename
            )  # Remove the file from the server since it is no longer needed
        source_type = ""
        file_source = ""
        file_chunks = []
        file_vectors = []

    return source_type, file_source, file_chunks, file_vectors


def embed_url(url):  # Create embeddings for a url
    source_type = "url"  # To help distinguish between local/URL files and URLs
    url_source = ""  # Source of the url
    url_chunks = []  # List of url chunks (for the url)
    url_vectors = []  # List of list of url embeddings (for each chunk)
    filename = ""  # Filename of the url if it is a file

    try:
        if validators.url(url, public=True):  # Verify url is a valid and public
            response = requests.get(url)  # Get the url info
            header = response.headers["Content-Type"]  # Get the header of the url
            is_application = (
                header.split("/")[0] == "application"
            )  # Check if the url is a file

            if is_application:  # If url is a file, call embed_file on the file
                filetype = header.split("/")[1]  # Get the filetype
                url_parts = url.split("/")  # Get the parts of the url
                filename = str(
                    "./"
                    + " ".join(
                        url_parts[:-1] + [url_parts[-1].split(".")[0]]
                    )  # Replace / with whitespace in the filename to avoid issues with the file path and remove the file extension since it may not match the actual filetype
                    + "."
                    + filetype
                )  # Create the filename
                with requests.get(
                    url, stream=True
                ) as stream_response:  # Download the file
                    stream_response.raise_for_status()
                    with open(filename, "wb") as file:
                        for chunk in stream_response.iter_content(
                            chunk_size=download_chunk_size
                        ):
                            file.write(chunk)
                return embed_file(filename)  # Embed the file
            else:  # If url is a webpage, use BeautifulSoup to extract the text
                soup = bs4.BeautifulSoup(response.text)  # Create a BeautifulSoup object
                extracted_text = (
                    soup.get_text()  # Extract the text from the webpage
                    .encode("ascii", "ignore")  # Remove non-ascii characters
                    .decode()  # Convert back to string
                )
                if not extracted_text:  # If the webpage is empty
                    raise Exception
                url_source = url
                url_chunks = split_into_many(
                    extracted_text
                )  # Split the text into chunks
                url_vectors = [
                    x["embedding"] for x in embed(url_chunks[-1])
                ]  # Embed the chunks
        else:  # If url is not valid or public
            raise Exception
    except Exception:  # If the url cannot be extracted, return empty values
        source_type = ""
        url_source = ""
        url_chunks = []
        url_vectors = []

    return source_type, url_source, url_chunks, url_vectors


def get_most_relevant(
    prompt_embedding, sources_embeddings
):  # Get which sources/chunks are most relevant to the prompt
    sources_indices = []  # List of indices of the most relevant sources
    sources_cosine_sims = []  # List of cosine similarities of the most relevant sources

    for (
        source_embeddings
    ) in (
        sources_embeddings
    ):  # source_embeddings contains all the embeddings of each chunk in a source
        cosine_sims = np.array(
            (source_embeddings @ prompt_embedding)
            / (
                np.linalg.norm(source_embeddings, axis=1)
                * np.linalg.norm(prompt_embedding)
            )
        )  # Calculate the cosine similarity between the prompt and each chunk's vector
        # Get the indices of the most relevant chunks: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        num_chunks = min(
            num_citations, len(cosine_sims)
        )  # In case there are less chunks than num_citations
        indices = np.argpartition(cosine_sims, -num_chunks)[
            -num_chunks:
        ]  # Get the indices of the most relevant chunks
        indices = indices[np.argsort(cosine_sims[indices])]  # Sort the indices
        cosine_sims = cosine_sims[
            indices
        ]  # Get the cosine similarities of the most relevant chunks
        sources_indices.append(indices)  # Add the indices to sources_indices
        sources_cosine_sims.append(
            cosine_sims
        )  # Add the cosine similarities to sources_cosine_sims

    # Use sources_indices and sources_cosine_sims to get the most relevant sources/chunks
    indexes = []
    max_cosine_sims = []
    for source_idx in range(len(sources_indices)):  # For each source
        for chunk_idx in range(len(sources_indices[source_idx])):  # For each chunk
            sources_chunk_idx = sources_indices[source_idx][
                chunk_idx
            ]  # Get the index of the chunk
            similarity = sources_cosine_sims[source_idx][
                chunk_idx
            ]  # Get the cosine similarity of the chunk
            if len(max_cosine_sims) < num_citations:  # If max_values is not full
                indexes.append(
                    [source_idx, sources_chunk_idx]
                )  # Add the source/chunk index pair to indexes
                max_cosine_sims.append(
                    similarity
                )  # Add the cosine similarity to max_values
            elif len(max_cosine_sims) == num_citations and similarity > min(
                max_cosine_sims
            ):  # If max_values is full and the current cosine similarity is greater than the minimum cosine similarity in max_values
                indexes.append(
                    [source_idx, sources_chunk_idx]
                )  # Add the source/chunk index pair to indexes
                max_cosine_sims.append(
                    similarity
                )  # Add the cosine similarity to max_values
                min_idx = max_cosine_sims.index(
                    min(max_cosine_sims)
                )  # Get the index of the minimum cosine similarity in max_values
                indexes.pop(
                    min_idx
                )  # Remove the source/chunk index pair at the minimum cosine similarity index in indexes
                max_cosine_sims.pop(
                    min_idx
                )  # Remove the minimum cosine similarity in max_values
            else:  # If max_values is full and the current cosine similarity is less than the minimum cosine similarity in max_values
                pass
    return indexes


def process_source(
    source, source_type
):  # Process the source name to be used in a message, since URL files are processed differently
    return (
        source if source_type == "file" else source.replace(" ", "/")
    )  # In case this is a URL, reverse what was done in embed_url


#   TODO: Find better way to create/store messages instead of everytime a new question is asked
def ask():  # Ask a question
    messages = [
        {
            "role": "system",
            "content": str(
                "You are a helpful chatbot that answers questions a user may have about a topic. "
                + "Sometimes, they may give you external data sources from which you can use as needed. "
                + "They will give them to you in the following way:\n"
                + "the source's name\n"
                + "the relevant text from the source\n\n\n\n\n\n\n\n\n\n"  # Multiple newlines to make sure it's understandable which sources are which
                + "...\n"
                + "You can use this data to answer the user's questions or to ask the user questions.\n"
            ),
        },
        {"role": "user", "content": st.session_state["questions"][0]},
    ]  # Add the system's introduction message and the user's first question to messages
    show_message(
        st.session_state["questions"][0],
        is_user=True,
        key=str(uuid.uuid4()),
        avatar_style=user_avatar_style,
    )  # Display user's first question

    if (
        len(st.session_state["questions"]) > 1 and st.session_state["answers"]
    ):  # If this is not the first question
        for interaction, message in enumerate(
            [
                message
                for pair in zip(
                    st.session_state["answers"], st.session_state["questions"][1:]
                )
                for message in pair
            ]  # Get the messages from the previous conversation in the order of [answer, question, answer, question, ...]: https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
        ):
            if interaction % 2 == 0:  # If the message is an answer
                messages.append(
                    {"role": "assistant", "content": message}
                )  # Add the answer to messages
                show_message(
                    message,
                    key=str(uuid.uuid4()),
                    avatar_style=assistant_avatar_style,
                )  # Display the answer
            else:  # If the message is a question
                messages.append(
                    {"role": "user", "content": message}
                )  # Add the question to messages
                show_message(
                    message,
                    is_user=True,
                    key=str(uuid.uuid4()),
                    avatar_style=user_avatar_style,
                )  # Display the question

    if (
        st.session_state["sources_types"]
        and st.session_state["sources"]
        and st.session_state["chunks"]
        and st.session_state["vectors"]
    ):  # If there are sources that were uploaded
        prompt_embedding = np.array(
            embed(messages[0]["content"] + '\n' + st.session_state["questions"][-1])[0]["embedding"]
        )  # Embed the instruction messager and the last question
        indexes = get_most_relevant(
            prompt_embedding, st.session_state["vectors"]
        )  # Get the most relevant chunks
        if indexes:  # If there are relevant chunks
            messages[-1]["content"] += str(
                "Here are some sources that may be helpful:\n"
            )  # Add the sources to the last message
            for ind in indexes:
                source_idx, chunk_idx = ind[0], ind[1]  # Get the source and chunk index
                messages[-1]["content"] += str(
                    process_source(
                        st.session_state["sources"][source_idx],
                        st.session_state["sources_types"][source_idx],
                    )
                    + "\n"
                    + st.session_state["chunks"][source_idx][chunk_idx]  # Get the chunk
                    + "\n\n\n\n\n\n\n\n\n\n"  # Multiple newlines to make sure it's understandable which sources are which
                )

    while (
        get_num_tokens("\n".join([message["content"] for message in messages]))
        > llm_context_window
    ):  # If the context window is too large
        if (
            len(messages) == 2
        ):  # If there is only the introduction message and the user's most recent question
            max_tokens_left = llm_context_window - get_num_tokens(
                messages[0]["content"]
            )  # Get the maximum number of tokens that can be present in the question
            messages[1]["content"] = messages[1]["content"][
                :max_tokens_left
            ]  # Truncate the question, from https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them 4 chars ~= 1 token, but it isn't certain that this is the case, so we will just truncate the question to max_tokens_left characters to be safe
        else:  # If there are more than 2 messages
            messages.pop(1)  # Remove the oldest question
            messages.pop(2)  # Remove the oldest answer

    answer = openai.ChatCompletion.create(model=llm_model, messages=messages)[
        "choices"
    ][0]["message"][
        "content"
    ]  # Get the answer from the chatbot
    st.session_state["answers"].append(answer)  # Add the answer to answers
    show_message(
        st.session_state["answers"][-1],
        key=str(uuid.uuid4()),
        avatar_style=assistant_avatar_style,
    )  # Display the answer


# Main function, defines layout of the app
def main():
    # Initialize session state variables
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "answers" not in st.session_state:
        st.session_state["answers"] = []
    if "sources_types" not in st.session_state:
        st.session_state["sources_types"] = []
    if "sources" not in st.session_state:
        st.session_state["sources"] = []
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "vectors" not in st.session_state:
        st.session_state["vectors"] = []

    st.title("CacheChat :money_with_wings:")  # Title
    st.markdown(
        "Check out the repo [here](https://github.com/andrewhinh/CacheChat) and notes on using the app [here](https://github.com/andrewhinh/CacheChat#notes)."
    )  # Link to repo

    uploaded_files = st.file_uploader(
        "Choose file(s):", accept_multiple_files=True, key="files"
    )  # File upload section
    if uploaded_files:  # If (a) file(s) is/are uploaded, create embeddings
        with st.spinner("Processing..."):  # Show loading spinner
            for uploaded_file in uploaded_files:
                if not (
                    uploaded_file.name in st.session_state["sources"]
                ):  # If the file has not been uploaded, process it
                    with open(uploaded_file.name, "wb") as file:  # Save file to disk
                        file.write(uploaded_file.getbuffer())
                    source_type, file_source, file_chunks, file_vectors = embed_file(
                        uploaded_file.name
                    )  # Embed file
                    if (
                        not source_type
                        and not file_source
                        and not file_chunks
                        and not file_vectors
                    ):  # If the file is invalid
                        st.error("Invalid file(s). Please try again.")
                    else:  # If the file is valid
                        st.session_state["sources_types"].append(source_type)
                        st.session_state["sources"].append(file_source)
                        st.session_state["chunks"].append(file_chunks)
                        st.session_state["vectors"].append(file_vectors)

    with st.form(key="url", clear_on_submit=True):  # form for question input
        uploaded_url = st.text_input(
            "Enter a URL:",
            placeholder="https://www.africau.edu/images/default/sample.pdf",
        )  # URL input text box
        upload_url_button = st.form_submit_button(label="Add URL")  # Add URL button
    if upload_url_button and uploaded_url:  # If a URL is entered, create embeddings
        with st.spinner("Processing..."):  # Show loading spinner
            if not (
                uploaded_url in st.session_state["sources"]  # Non-file URL in sources
                or "./" + uploaded_url.replace("/", " ")  # File URL in sources
                in st.session_state["sources"]
            ):  # If the URL has not been uploaded, process it
                source_type, url_source, url_chunks, url_vectors = embed_url(
                    uploaded_url
                )  # Embed URL
                if (
                    not source_type
                    and not url_source
                    and not url_chunks
                    and not url_vectors
                ):  # If the URL is invalid
                    st.error("Invalid URL. Please try again.")
                else:  # If the URL is valid
                    st.session_state["sources_types"].append(source_type)
                    st.session_state["sources"].append(url_source)
                    st.session_state["chunks"].append(url_chunks)
                    st.session_state["vectors"].append(url_vectors)

    st.divider()  # Create a divider between the uploads and the chat

    input_container = (
        st.container()
    )  # container for inputs/uploads, https://docs.streamlit.io/library/api-reference/layout/st.container
    response_container = (
        st.container()
    )  # container for chat history, https://docs.streamlit.io/library/api-reference/layout/st.container

    with input_container:
        with st.form(key="question", clear_on_submit=True):  # form for question input
            uploaded_question = st.text_input(
                "Enter your input:",
                placeholder="e.g: Summarize the research paper in 3 sentences.",
                key="input",
            )  # question text box
            uploaded_question_button = st.form_submit_button(
                label="Send"
            )  # send button

    with response_container:
        if (
            uploaded_question_button and uploaded_question
        ):  # if send button is pressed and text box is not empty
            with st.spinner("Thinking..."):  # show loading spinner
                st.session_state["questions"].append(
                    uploaded_question
                )  # add question to questions
                ask()  # ask question to chatbot


if __name__ == "__main__":
    main()
