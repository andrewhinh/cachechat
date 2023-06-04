# [CacheChat](https://cachechat.pagekite.me/)

Chat about your data, simple as that. Try out the website [here](https://cachechat.pagekite.me/).

https://github.com/andrewhinh/CacheChat/assets/40700820/ba773758-b0e4-4b5a-a436-bd4426791978

## Notes

- You don't need to upload anything to chat. It just helps the AI out, and who doesn't want to do that?
- You can upload as many files/URLs as you want at any point in the conversation.
- A list of accepted file/URL file formats can be found [here](https://textract.readthedocs.io/en/stable/#currently-supporting). Keep in mind that it sometimes won't be able to extract the text from the file (a blurry image for example).
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
    git clone https://github.com/andrewhinh/cachechat.git
    cd cachechat
    conda env update --prune -f environment.yml
    conda activate cachechat
    pip install -r requirements.txt
    export PYTHONPATH=.
    echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc (or ~/.zshrc)
    # If on Windows, the last two lines probably won't work. Check out this guide for more info: https://datatofish.com/add-python-to-windows-path/
    ```

3. Using `.env.template` as reference, create a `.env` file with your [OpenAI API key](https://beta.openai.com/account/api-keys), and reactivate the conda environment:

    ```bash
    conda activate cachechat
    ```

4. Run the application using streamlit:

   ```bash
   streamlit run app.py
   ```
