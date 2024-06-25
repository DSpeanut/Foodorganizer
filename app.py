from transformers import AutoTokenizer, pipeline
from Langchain_prompt import vectorstore_transformation, prompt_setup, llama27b_chat, langchain_output
import gradio as gr


def process(audio):
    def transcribe(audio):
        speech_recog = pipeline(task = "automatic-speech-recognition", model = 'openai/whisper-small')
        text = speech_recog(audio)["text"]
        return text
    
    processed_text = transcribe(audio)
    vectorstore = vectorstore_transformation(processed_text)
    prompt = prompt_setup()
    llm_model = llama27b_chat(0.9, 512, 0.9)
    output = langchain_output(vectorstore, prompt,llm_model)
    return output

app = gr.Interface(
    title = "Food ingredient organizer",
    theme = "gradio/monochrome",
    description = "<h2><center>Tell us what you have in the fridge, I'll organize it for you 🤗</center></h2> \
        <br><h3> Recording Example: I have 4 sweet potatoes, 500 grams of minced pork, and 2 chicken breasts. </h3>",
    fn=process, 
    inputs=gr.Audio(sources="microphone", type="filepath"), 
    outputs="text",live=True)

if __name__ == "__main__":
    app.launch()