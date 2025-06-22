# English-Malay Translator
This project implements an English-to-Malay Translator using the pretrained mesolitica/t5-base-standard-bahasa-cased T5 (Text-to-Text Transfer Transformer) model. The application is built with Streamlit, providing a user-friendly web interface for translating English sentences into Malay, with the ability to view translation history. The model was evaluated using a parallel dataset from the Asian Languages Treebank (ALT) with metrics such as BLEU, ChrF, TER, and METEOR.

## Features
Translation: Converts English sentences to Malay bidirectionally using the mesolitica/t5-base-standard-bahasa-cased model.
Translation History: Show previous translations for user reference.

## Installation
1. Clone the repository:
* git clone https://github.com/choo06/English-Malay-Translator.git
* cd English-Malay-Translator
2. Install dependencies:
* pip install -r requirements.txt
3. Run the Streamlit app:
* streamlit run app.py
4. Open your browser and navigate to http://localhost:8501

## Getting Started
* Access the Website: Open your browser and enter this URL https://tms4043-ai-project.streamlit.app