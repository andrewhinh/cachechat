# [CacheChat](https://cachechat.pagekite.me/)

Chat about your data, simple as that. Try out the website [here](https://cachechat.pagekite.me/).

## Notes

- You don't need to upload anything to chat. It just helps the AI out, and who doesn't want to do that?
- You can upload new files/URLs at any point in the conversation.
- Once a file/URL is uploaded, it'll only be removed if you restart the conversation by refreshing the page.

## Setup

1. Install conda if necessary:

    ```bash
    # Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
    # If on Windows, install chocolately: https://chocolatey.org/install. Then, run:
    # choco install make
    ```

2. Create the conda environment and install pip packages locally:

    ```bash
    git clone https://github.com/andrewhinh/CacheChat.git
    cd CacheChat
    conda env update --prune -f environment.yml
    conda activate cc
    pip install -r requirements.txt
    export PYTHONPATH=.
    echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
    ```

3. Using `.env.template` as reference, create a `.env` file with your [OpenAI API key](https://beta.openai.com/account/api-keys), and reactivate the conda environment:

    ```bash
    conda activate CacheChat
    ```

4. Run the application using streamlit:

   ```bash
   streamlit run app.py
   ```
