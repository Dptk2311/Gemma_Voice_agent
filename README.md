# README.md

## Overview

This project is a voice-powered Q&A assistant for a pizza restaurant, using AI models for speech recognition, language understanding, and text-to-speech. It leverages OpenAI Whisper for speech-to-text, Ollama LLM for language generation, Google Cloud TTS for speech synthesis, and Chroma vector database for semantic search over restaurant reviews.

- [`test3.py`](test3.py): Main script for voice-based Q&A with real-time speech input/output.
- [`vector.py`](vector.py): Loads restaurant reviews, builds a vector database, and provides a retriever for semantic search.

## Features

- Record your question via microphone
- Transcribe speech to text using Whisper
- Retrieve relevant restaurant reviews using vector search
- Generate concise answers using an Ollama LLM
- Speak answers aloud using Google Cloud TTS

## Setup

### 1. Clone the repository

```sh
git clone https://github.com/Dptk2311/Gemma_Voice_agent.git
cd Gemma_Voice_agent
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Download Ollama models

You must have [Ollama](https://ollama.com/) installed and running.

```sh
ollama pull gemma3:1b
ollama pull mxbai-embed-large
```

### 4. Prepare data

Ensure you have the file `realistic_restaurant_reviews.csv` in the project directory. This should contain columns: `Title`, `Review`, `Rating`, `Date`.

### 5. Google Cloud TTS

- Create a Google Cloud project and enable the Text-to-Speech API.
- Download your service account JSON credentials.
- Set the environment variable before running the script:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

(On Windows, use `set` instead of `export`.)

## Usage

### First-time setup

Run [`vector.py`](vector.py) once to initialize the vector database:

```sh
python vector.py
```

### Main voice assistant

Run the assistant:

```sh
python test3.py
```

Follow the prompts to record your question and hear the AI's spoken answer.

## File Descriptions

- [`test3.py`](test3.py): Main entry point. Handles audio recording, transcription, retrieval, LLM answering, and TTS playback.
- [`vector.py`](vector.py): Loads reviews from CSV, builds/loads a Chroma vector database, and exposes a retriever for semantic search.

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## Troubleshooting

- **Ollama errors**: Ensure Ollama is running and the required models are pulled.
- **Google TTS errors**: Make sure your credentials are set and the API is enabled.
- **Microphone/audio issues**: Check your device permissions and audio drivers.

---

## License

For educational and demonstration purposes only.
