# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image

# Function part
# List of words not suitable for kids' stories
UNSAFE_WORDS = [
    "smoking", "cigarette", "alcohol", "beer", "wine", "drug",
    "gun", "knife", "blood", "kill", "dead", "fight", "war",
    "scary", "horror", "monster", "ghost", "violent", "weapon"
]

SAFE_CAPTIONS = [
    "a happy child eating a delicious cookie in a sunny garden",
    "a friendly person reading a book under a big tree",
    "a cheerful girl playing with colorful flowers in a park"
]

def filter_caption(caption):
    caption_lower = caption.lower()
    for word in UNSAFE_WORDS:
        if word in caption_lower:
            # Replace with a kid-friendly generic caption
            safe_caption = random.choice(SAFE_CAPTIONS)
            return safe_caption, True
    return caption, False
    
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    prompt = f"Once upon a time, {text}. "
    story_results = story_pipe(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1
    )
    story = story_results[0]["generated_text"]
    return story

# text2audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)
    return audio_data

# Main part
st.set_page_config(
    page_title="Kid's Storytelling Session 🌈",
    page_icon="🧸",
    layout="centered"
)
st.title("📚 Kid's Storytelling App Demo✨")
st.header("Welcome to Kid's Storytelling Session! 🎉")
st.subheader("Let's turn your picture into a story!")
st.markdown(
    "🖼️ **Upload a image** → 📝 **Get a story** → 🔊 **Listen to it**"
)
uploaded_file = st.file_uploader("Select an Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text (Using the function)
    with st.spinner("🔍 Processing picture..."):
        scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    safe_scenario, was_filtered = filter_caption(scenario)
    if was_filtered:
        st.warning("🛡️ I made the description more fun for kids!")
    st.success(f"🖼️ **Scenario:** {safe_scenario}")

    # Stage 2: Text to Story (Inline)
    with st.spinner("✍️ Generating a story for you..."):
        story = text2story(scenario)
    st.info(f"📖 **Story:**\n\n{story}")

    # Stage 3: Story to Audio (Inline)
    with st.spinner("🎤 Getting ready to read story..."):
        audio_data = text2audio(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>"
    "Made with ❤️ for little storytellers everywhere"
    "</p>",
    unsafe_allow_html=True
)
