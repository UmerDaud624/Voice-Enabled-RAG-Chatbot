Voice-Enabled RAG Chatbot

A simple RAG chatbot built with LangGraph, Gemini, ChromaDB, MongoDB, Tavily, Whisper, and Gradio.

The chatbot can answer questions from a MapReduce PDF, search the web when needed, and accept both text and voice input.

Features
RAG using ChromaDB
Google Gemini for responses
LangGraph agent with tool calling
Tavily web search
Voice input using Faster-Whisper
MongoDB for chat history
Multiple chat sessions
Gradio web interface
Tech Stack
Python
LangGraph / LangChain
Google Gemini
ChromaDB
MongoDB
Tavily
Faster-Whisper
PyAudio
Gradio
Setup

Clone the repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo


Create a virtual environment:

python -m venv venv
venv\Scripts\activate


Install the dependencies:

pip install -r requirements.txt


Create a .env file:

GEMINI_API_KEY=your_gemini_api_key
MONGODB_URL=your_mongodb_url
TAVILY_API_KEY=your_tavily_api_key


Place the MapReduce PDF in the project folder:

mapreduce-osdi04.pdf


Then run:

python app.py


Open the Gradio URL shown in the terminal.

Example Questions
What is MapReduce and how does it work?

Explain the key components of MapReduce.

What are the advantages of using MapReduce?

How does fault tolerance work in MapReduce?

Project Structure
project/
├── app.py
├── mapreduce-osdi04.pdf
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
