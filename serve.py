import shlex
import subprocess
from pathlib import Path

import modal

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"


NAME = "cachechat"

PARENT_PATH = Path(__file__).parent
SECRETS = [modal.Secret.from_dotenv(path=PARENT_PATH)]
PYTHON_VERSION = "3.12"

IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(  # add Python dependencies
        "beautifulsoup4==4.12.3",
        "modal==0.72.11",
        "nltk==3.9.1",
        "numpy==2.2.1",
        "openai==1.59.7",
        "python-dotenv==1.0.1",
        "streamlit-chat==0.1.1",
        "streamlit==1.41.1",
        "textract==1.5.0",
        "tiktoken==0.8.0",
        "validators==0.34.0",
    )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

MINUTES = 60  # seconds
APP_TIMEOUT = 1 * MINUTES
APP_SCALEDOWN_WINDOW = 15 * MINUTES  # max
APP_ALLOW_CONCURRENT_INPUTS = 1000  # max

APP_NAME = f"{NAME}"
app = modal.App(name=APP_NAME)

PORT = 8000


@app.function(
    image=IMAGE,
    secrets=SECRETS,
    timeout=APP_TIMEOUT,
    scaledown_window=APP_SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=APP_ALLOW_CONCURRENT_INPUTS)
@modal.web_server(PORT)
def modal_get():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port {PORT} --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
