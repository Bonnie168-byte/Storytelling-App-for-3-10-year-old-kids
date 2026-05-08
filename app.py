# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image
import time

# Function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_text = pipeline("text-generation", model="google/gemma-2-9b-it")
    return story_text

# text2audio
def text2audio(story_text):
    audio_data = ""     # to be completed
    return audio_data

# Main part
st.set_page_config(
    page_title="Magic Story Time 🌈",
    page_icon="🧸",
    layout="centered"
)
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text (Using the function)
    st.text('Processing img2text...')
    
    image_to_text_model = pipeline("image-to-text", 
                               model="Salesforce/blip-image-captioning-base")
    scenario = image_to_text_model(uploaded_file.name)[0]["generated_text"]
  
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (Inline)
    st.text('Generating a story...')
  
    story_pipe = pipeline("text-generation", 
                          model="roneneldan/TinyStories-1M")
    story_results = story_pipe(scenario)
    story = story_results[0]['generated_text']
  
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (Inline)
    st.text('Generating audio data...')
    audio_pipe = pipeline("text-to-audio", 
                          model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
