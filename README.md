This is Slant, an AI chatbot built with Streamlit and originally using a DeepSeek API, with a resources folder, that utilizes Retrieval-Augmented Generation to draw on provided material when generating responses.
It has many features, such as persistent chat history, message editing, and chat deletion.
Requires Python 3.8+ and your own DeepSeek API key; if using a different API, code changes are necessary.

Installation:
1. Clone Repository
2. Install dependencies: pip install streamlit openai langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
3. Create a .streamlit/secrets.toml file and add your API key: deepseek-key = "your-api-key-here"
4. Add any .txt resource file to a folder named resources in the project directory
5. Run with: Streamlit run main.py

Notes:
The first run takes more time, depending on how many files are in the resources folder. Once built, knowledge_index is saved locally, allowing for faster run times afterward.
Never share or commit your secrets.toml file.
