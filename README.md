# Food ingredient organizer

## Simple application to identify ingredients and its quantity or measurement utilizing Speech2text, LLM RAG, and Gradio from audio recording. 

This project is a simple example how to apply LLM RAG to build a solution that identify food ingredients and its quantity or measurement. This project includes a couple of components;

* Speech2text model : openai/whisper-small
* LLM : Llama (llama-2-7b-chat.Q2_K.gguf version downloaded locally), llama.cpp
* RAG tools : Langchain
* Gradio : Quick web application platform

## How to install this example project and run it

1. Clone this project
2. Download llama 2-7b-chat-Q2_K.gguf file here and replace with model_download_path: https://huggingface.co/TheBloke/Llama-2-7B-GGUF
3. Adjust the prompt as you wish!