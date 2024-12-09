# How to Use

1. Install the required library
   ```
   pip install -r requirements.txt
   ```
2. Create vector database using notebook [create_vector_db.ipynb](/llm-chatbot/create_vector_db.ipynb)
3. Download model using Ollama
   ```
   ollama pull qwen2.5:14b-instruct-q5_K_M
   ```
4. Run chatbot app
   ```
   nohup python app.py
   ```