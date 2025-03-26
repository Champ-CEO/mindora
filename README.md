# Mindora

Local chatbot with conversation history and memory capabilities with persistence using SQLite

![Mindora interface](.github/banner.png)

Features:

- Local chatbot that acts as a wellness coach
- Conversation history stored in database (SQLite)
- Memory management: remembers user preferences and past interactions
- Uses Groq API for LLM inference

## Workflow components

- [`Load memory`](mindora/chatbot.py): loads stored memories for the current user
- [`Generate response`](mindora/chatbot.py): generates a response to the user query using the conversation history and memories
- [`Save new memory`](mindora/chatbot.py): decide whether to save the new memory based on the user query

## Install

Make sure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone git@github.com:mlexpertio/mindora.git .
cd mindora
```

Install Python:

```bash
uv python install 3.12.8
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Install package in editable mode:

```bash
uv pip install -e .
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Groq API Configuration

Mindora uses Groq API for LLM inference. You'll need to get an API key from [Groq console](https://console.groq.com/keys).

Rename the `.env.example` file to `.env` and add your API key inside:

```bash
mv .env.example .env
```

Edit the `.env` file and set your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

The application uses the following Groq models:
- Chat model (general tasks): `llama-3.3-70b-versatile`
- Tool model (complex tasks): `deepseek-r1-distill-llama-70b`

These settings can be adjusted in the [`config.py`](mindora/config.py) file if needed.

## Run the Streamlit app

Run the app:

```bash
streamlit run app.py