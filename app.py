import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

torch.classes.__path__ = [] 

st.set_page_config(
    page_title="Translator (EN ↔ BM)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model():
    model_name = "mesolitica/t5-base-standard-bahasa-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    return tokenizer, model

# Call the cached function to load the model
tokenizer, model = load_model()

# --- Languages Available ---
LANG_EN = "English"
LANG_BM = "Bahasa Malaysia"
available_languages = [LANG_EN, LANG_BM]

# --- Streamlit UI ---
st.title("English ↔ Bahasa Malaysia Translator")

# --- Language Selection and Swap ---
col1_lang, col2_swap, col3_lang = st.columns([1, 0.2, 1])

with col1_lang:
    if 'source_lang_index' not in st.session_state:
        st.session_state['source_lang_index'] = 0 # Default to English

    source_lang = st.selectbox(
        "Source Language",
        available_languages,
        index=st.session_state['source_lang_index'],
        key="source_lang_select"
    )

with col2_swap:
    st.write("") 
    st.write("") 
    if st.button("↔", help="Swap languages", key="swap_button"):
        st.session_state['source_lang_index'] = 1 - st.session_state['source_lang_index']
        st.rerun()

with col3_lang:
    target_lang = LANG_BM if source_lang == LANG_EN else LANG_EN
    
    st.text_input(
        label="Target Language",
        value=target_lang,
        disabled=True,
        key="target_lang"
    )

st.markdown("---") 

# Create two columns for the input and output text areas
input_col, output_col = st.columns(2)

# Input Text Area
with input_col:
    input_placeholder = f"Enter text in {source_lang}:"
    if source_lang == LANG_EN:
        default_input_text = "I love you."
    else: 
        default_input_text = "Saya sayang awak."

    input_text = st.text_area(
        input_placeholder,
        default_input_text,
        height=200, 
        key="input_text_area"
    )

# Output Text Area 
with output_col:
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    
    st.text_area(
        f"{target_lang} Translation:",
        st.session_state.translated_text,
        height=200,
        disabled=True,
        key="output_text_area"
    )

# Dynamic T5 Prefix 
t5_prefix = ""
if source_lang == LANG_EN and target_lang == LANG_BM:
    t5_prefix = "terjemah Inggeris ke Melayu:"
elif source_lang == LANG_BM and target_lang == LANG_EN:
    t5_prefix = "terjemah Melayu ke Inggeris:"
else:
    st.error("Invalid language combination for translation. This should not happen.")
    st.stop()

# Translate Button 
if st.button("Translate", key="translate_button", use_container_width=True):
    if not input_text:
        st.warning(f"Please enter some text in {source_lang} to translate.")
    else:
        try:
            processed_input = f"{t5_prefix} {input_text}"

            inputs = tokenizer(processed_input, return_tensors="pt", max_length=512, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_length": 150,
                "num_beams": 5,
                "early_stopping": True
            }

            outputs = model.generate(**generation_kwargs)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.session_state.translated_text = translated_text
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during translation: {e}")