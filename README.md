# AI Invoice Extraction Service

FastAPI app that:
- Proxies chat requests to OpenAI (`/v1/chat/completions`)
- Extracts invoice fields (`/v1/prefill`) using **GPT-5** with regex fallback

## Setup

```bash
git clone <your-repo-url>
cd AI-Server
python -m venv .venv
. .venv/Scripts/activate   # (Windows)
# source .venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt

