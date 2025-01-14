# CacheChat

Chat about your data, simple as that.

<https://github.com/andrewhinh/CacheChat/assets/40700820/ba773758-b0e4-4b5a-a436-bd4426791978>

## Notes

- You can upload as many files/URLs as you want at any point in the conversation.
- A list of accepted file/URL file formats can be found [here](https://textract.readthedocs.io/en/stable/#currently-supporting). Keep in mind that it sometimes won't be able to extract the text from the file (a blurry image for example).
- Once a file/URL is uploaded, it'll only be removed if you restart the conversation by refreshing the page.

## Setup

1. Run setup:

    ```bash
    uv sync
    ```

2. Using `.env.template` as reference, create a `.env` file with your [OpenAI API key](https://beta.openai.com/account/api-keys).

3. Run the application using streamlit:

   ```bash
    uv run streamlit run app.py
   ```

4. Deploy the app on Modal:

   ```bash
   modal deploy --env main serve.py
   ```
