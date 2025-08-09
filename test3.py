import sounddevice as sd
import whisper
import numpy as np
import scipy.io.wavfile as wav # Still needed for saving recorded input
import os
# from TTS.api import TTS # Removed
from google.cloud import texttospeech # Added
from vector import retriever


from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Assuming 'vector' and 'retriever' are custom modules you have
# For demonstration, I'll create a dummy retriever. Replace with your actual retriever.
"""
class DummyRetriever:
    def invoke(self, query):
        print(f"DummyRetriever invoked with: {query}")
        # Simulate a list of Document-like objects or strings
        # from langchain_core.documents import Document
        # return [Document(page_content="Sample review: The pizza was amazing."), Document(page_content="Service was quick.")]
        return "Sample review: The pizza was amazing and the service was quick."
"""

#retriever = DummyRetriever()

# --- Constants ---
RECORD_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000  # Wavenet voices often use 24kHz for good quality

# Initialize STT model
stt_model = whisper.load_model("base")

# LangChain setup
model_name_ollama = "gemma3:1b" # Adjust if needed
try:
    model = OllamaLLM(model=model_name_ollama)
except Exception as e:
    print(f"Error initializing OllamaLLM with model '{model_name_ollama}': {e}")
    print("Please ensure Ollama is running and the model is available ('ollama list').")
    exit()

template = """
You are an expert in answering questions about a pizza restaurant.

You have access to a collection of customer reviews about the restaurant.
You will answer the question based on the reviews provided in maximum 100 words.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Record audio
def record_audio(filename="input.wav", duration=5, fs=RECORD_SAMPLE_RATE):
    print("🎙️ Recording...")
    # Ensure correct dtype for WAV and Whisper
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, recording) # Still save the recorded input for transcription
    print("✅ Recorded")

# Transcribe audio to text
def transcribe(filename="input.wav"):
    print("🧠 Transcribing...")
    try:
        result = stt_model.transcribe(filename)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

# Speak output text directly using Google Cloud TTS and sounddevice
def speak(text):
    print("🗣️ Speaking with Google TTS (direct playback)...")
    try:
        # Instantiates a client
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Wavenet-C", # High-quality Wavenet voice
        )

        # Select the type of audio encoding and explicitly set sample rate
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=TTS_SAMPLE_RATE # Important for playback
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Convert the binary audio content to a NumPy array
        # LINEAR16 means signed 16-bit PCM audio
        audio_array = np.frombuffer(response.audio_content, dtype=np.int16)

        # Play the audio using sounddevice
        print(f"Playing audio at {TTS_SAMPLE_RATE} Hz...")
        sd.play(audio_array, samplerate=TTS_SAMPLE_RATE)
        sd.wait()  # Wait for playback to complete
        print("✅ Playback finished")

    except Exception as e:
        print(f"Error during Google TTS processing or direct playback: {e}")
        print("Ensure you have set up Google Cloud authentication correctly (GOOGLE_APPLICATION_CREDENTIALS).")
        print("Also check your audio output device.")

# Chat loop
while True:
    print("\n\n-------------------------------")
    mode = input("🎤 Press Enter to talk (or type 'q' to quit): ")
    if mode.strip().lower() == "q":
        print("Exiting...")
        break
    
    input_audio_file = "input.wav" # Still used for the recording
    record_audio(filename=input_audio_file)
    question = transcribe(filename=input_audio_file)

    if not question.strip():
        print("⚠️ No question transcribed. Please try again.")
        continue
    
    print(f"📝 You asked: {question}")

    try:
        reviews = retriever.invoke(question)
        # Ensure reviews is a string. Langchain components might return Document objects.
        if not isinstance(reviews, str):
            if hasattr(reviews, '__iter__') and all(hasattr(doc, 'page_content') for doc in reviews):
                reviews_text = "\n".join([doc.page_content for doc in reviews])
            elif hasattr(reviews, 'page_content'):
                reviews_text = reviews.page_content
            else:
                reviews_text = str(reviews)
        else:
            reviews_text = reviews

        result = chain.invoke({"reviews": reviews_text, "question": question})
        print(f"🤖 Bot: {result}")

        # Speak directly
        speak(result)

    except Exception as e:
        print(f"An error occurred in the main loop (retrieval/answering/speaking): {e}")