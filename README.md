# CacheChat

Chat with a file, simple as that.

## Setup

1. Install conda if necessary:

    ```bash
    # Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
    # If on Windows, install chocolately: https://chocolatey.org/install. Then, run:
    # choco install make
    ```

2. Create the conda environment locally:

    ```bash
    cd CacheChat
    make conda-update
    conda activate CacheChat
    pip install -r requirements.txt
    export PYTHONPATH=.
    echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
    ```

3. Sign up for an OpenAI account and get an API key [here](https://beta.openai.com/account/api-keys).

4. Populate a `.env` file with your API key in the format of `.env.template`, and reactivate the environment.

5. Run the application using streamlit:

   ```bash
   streamlit run app.py
   ```
