import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gradio as gr
import uuid
from tavily import TavilyClient
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch
import librosa
import threading
import queue

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from langchain_chroma import Chroma
from langchain_core.tools import tool
import google.generativeai as genai
from langgraph.prebuilt import ToolNode
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Voice Recognition Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MongoDB setup
url = os.getenv("MONGODB_URL")
client = MongoClient(url, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("✅ Pinged your deployment. Connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

# Set up database and collection
db = client["rag_chat_history"]
collection = db["conversations"]
voice_collection = db["voice_recordings"]

class GeminiEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-004"):
        self.model = model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            genai.embed_content(model=self.model, content=text, task_type="retrieval_document")["embedding"]
            for text in texts
        ]
    
    def embed_query(self, text: str) -> list[float]:
        return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")["embedding"]

pdf_path = "mapreduce-osdi04.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pages_split = text_splitter.split_documents(pdf_loader.load())

persist_directory = r"C:\Users\umerd\source\repos\ReAct agent NetSol"
collection_name = "mapreduce-osdi04"
embedding = GeminiEmbeddings()

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")
except Exception as e:
    logger.error(f"Error creating vector store: {e}")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name
    )
    print("Loaded existing ChromaDB vector store!")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the mapreduce-osdi04.
    """
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the mapreduce-osdi04."
        results = []
        for i, doc in enumerate(docs):
            content = doc.page_content[:1000]
            results.append(f"Document {i+1}:\n{content}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"Error in retriever_tool: {e}")
        return f"Error retrieving information: {str(e)}"

@tool
def tavily_search(query: str) -> str:
    """Performs a real-time web search using Tavily."""
    try:
        response = tavily_client.search(query, max_results=5)
        results = [f"Web Result {i+1} (Source: {r['url']}):\n{r['content'][:500]}..." for i, r in enumerate(response.get("results", []))]
        return "\n\n".join(results) if results else "No relevant web information found."
    except Exception as e:
        logger.error(f"Error in tavily_search: {e}")
        return f"Error performing web search: {str(e)}"

tools = [retriever_tool, tavily_search]
tools_dict = {tool.name: tool for tool in tools}

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_tokens=1000
).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

SYSTEM_PROMPT = """
You are an intelligent AI assistant designed to provide accurate and helpful responses.
Your primary knowledge base is the 'mapreduce-osdi04' PDF document,
which you should automatically consult using the retriever_tool for any questions related to technical terms or information.
Always use the retriever_tool first for such queries,
and only fall back to your general knowledge if the tool returns no relevant data or the query
is clearly unrelated to the document (e.g., current events, math problems, or broad knowledge topics).
When using the retriever_tool, cite the specific document sections (e.g., 'Document 1') in your answers.
If no relevant data is found, acknowledge it and attempt to answer based on general knowledge if possible,
clearly stating when you are relying on general knowledge rather than the document.
Maintain context from previous messages in the conversation, using the provided chat history to inform your responses,
while respecting the token limit. Strive to be concise, clear, and proactive in using your tools without requiring explicit user instructions."""

def call_llm(state: AgentState) -> AgentState:
    try:
        system_prompt = SystemMessage(content=SYSTEM_PROMPT)
        messages = list(state['messages'])
        session_id = getattr(call_llm, 'current_session_id', 'session_1')
        previous_chats = collection.find({"session_id": session_id}).sort("timestamp", -1).limit(10)
        historical_messages = []
        total_length = len(SYSTEM_PROMPT) + sum(len(str(msg.content)) for msg in messages)
        max_length = 100000

        for chat in previous_chats:
            user_msg = HumanMessage(content=chat["user_input"])
            ai_msg = HumanMessage(content=chat["ai_response"])
            user_len = len(str(user_msg.content))
            ai_len = len(str(ai_msg.content))
            if total_length + user_len + ai_len <= max_length:
                historical_messages.extend([user_msg, ai_msg])
                total_length += user_len + ai_len
            else:
                break

        messages = [system_prompt] + historical_messages[::-1] + messages
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.invoke(messages)
                return {'messages': [response]}
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    except Exception as e:
        logger.error(f"Error in call_llm: {e}")
        error_message = HumanMessage(content=f"I encountered an error: {str(e)}. Please try again or rephrase your question.")
        return {'messages': [error_message]}

def take_action(state: AgentState) -> AgentState:
    try:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
            if t['name'] not in tools_dict:
                print(f"\nTool: {t['name']} does not exist.")
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                try:
                    result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                    print(f"Result length: {len(str(result))}")
                except Exception as e:
                    logger.error(f"Error executing tool {t['name']}: {e}")
                    result = f"Error executing tool: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Tools Execution Complete. Back to the model!")
        return {'messages': results}
    except Exception as e:
        logger.error(f"Error in take_action: {e}")
        error_result = ToolMessage(tool_call_id="error", name="error", content=f"Tool execution error: {str(e)}")
        return {'messages': [error_result]}

def should_continue(state: AgentState):
    try:
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    except Exception as e:
        logger.error(f"Error in should_continue: {e}")
        return False

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()

class VoiceRecognizer:
    def __init__(self):
        self.whisper_model = None
        self.audio_interface = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.stream = None
        self.frames = []
        
    def initialize_models(self):
        try:
            print(f"🔧 Loading Whisper model on {DEVICE}...")
            self.whisper_model = WhisperModel("base", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
            self.audio_interface = pyaudio.PyAudio()
            print("✅ Voice recognition models loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error initializing voice models: {e}")
            return False
    
    def start_recording(self):
        """Start recording audio using the default device"""
        if not self.whisper_model or not self.audio_interface:
            return "Voice recognition not initialized", None
        
        if self.is_recording:
            return "Already recording", None
            
        try:
            self.is_recording = True
            self.frames = []
            
            # Open audio stream with default device
            self.stream = self.audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("🎤 Recording started...")
            
            # Start recording in a separate thread
            def record():
                while self.is_recording:
                    try:
                        data = self.stream.read(CHUNK, exception_on_overflow=False)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        self.frames.append(audio_data)
                    except Exception as e:
                        logger.error(f"Error during recording: {e}")
                        self.is_recording = False
                        break
            
            self.recording_thread = threading.Thread(target=record)
            self.recording_thread.start()
            return "Recording started", None
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.is_recording = False
            return f"Error starting recording: {str(e)}", None
    
    def stop_recording(self):
        """Stop recording and transcribe the audio"""
        if not self.is_recording:
            return "Not recording", None
            
        try:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            if not self.frames:
                return "No audio recorded", None
                
            # Process audio
            full_audio = np.concatenate(self.frames).astype(np.float32) / 32768.0
            if RATE != 16000:
                full_audio = librosa.resample(full_audio, orig_sr=RATE, target_sr=16000)
            
            print("🧠 Transcribing...")
            segments, _ = self.whisper_model.transcribe(full_audio, language="en")
            transcript = " ".join(segment.text for segment in segments).strip()
            
            if not transcript:
                return "No speech detected. Please try again.", None
                
            print(f"📝 Transcript: {transcript}")
            return transcript, None
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return f"Error processing audio: {str(e)}", None
    
    def cleanup(self):
        try:
            if self.is_recording and self.stream:
                self.is_recording = False
                self.stream.stop_stream()
                self.stream.close()
            if self.audio_interface:
                self.audio_interface.terminate()
        except Exception as e:
            logger.error(f"Error cleaning up audio: {e}")

voice_recognizer = VoiceRecognizer()

def get_chat_sessions():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$session_id",
                    "first_message": {"$first": "$user_input"},
                    "last_updated": {"$max": "$timestamp"},
                    "message_count": {"$sum": 1}
                }
            },
            {"$sort": {"last_updated": -1}},
            {"$limit": 50} 
        ]
        sessions = list(collection.aggregate(pipeline))
        session_choices = []
        for idx, session in enumerate(sessions, 1):
            session_id = session["_id"]
            first_msg = session["first_message"][:30] + "..." if len(session["first_message"]) > 30 else session["first_message"]
            last_updated = session["last_updated"].strftime("%Y-%m-%d %H:%M")
            msg_count = session["message_count"] // 2
            display_text = f"Chat {idx} - {first_msg} ({msg_count} msgs) [{last_updated}]"
            session_choices.append((display_text, session_id))
        return session_choices if session_choices else [("No previous chats", "")]
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        return [("Error loading chats", "")]

def load_chat_history(session_id):
    if not session_id or session_id == "":
        return [], session_id
    try:
        messages = list(collection.find({"session_id": session_id}).sort("timestamp", 1))
        if not messages:
            return [], session_id
        history = []
        for msg in messages:
            history.append({"role": "user", "content": msg["user_input"]})
            history.append({"role": "assistant", "content": msg["ai_response"]})
        logger.info(f"Loaded {len(messages)} message pairs for session {session_id}")
        return history, session_id
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return [], session_id

def delete_chat_session(session_id):
    if not session_id or session_id == "":
        return gr.update(choices=get_chat_sessions(), value=""), "", [], "No session selected to delete"
    
    try:
        # Delete the selected session from MongoDB
        result = collection.delete_many({"session_id": session_id})
        voice_result = voice_collection.delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages and {voice_result.deleted_count} voice recordings for session {session_id}")
        
        # Get updated list of chat sessions
        updated_sessions = get_chat_sessions()
        
        # If no sessions remain, reset to a new chat
        if not updated_sessions or updated_sessions[0][0] == "No previous chats":
            new_session_id = str(uuid.uuid4())
            initial_chat_entry = {
                "user_input": "New chat started",
                "ai_response": "Chat initialized",
                "timestamp": datetime.utcnow(),
                "session_id": new_session_id,
                "input_type": "system"
            }
            collection.insert_one(initial_chat_entry)
            logger.info(f"Created new session {new_session_id} as no sessions remain")
            updated_sessions = get_chat_sessions()
            return gr.update(choices=updated_sessions, value=new_session_id), "", [], ""
        
        # Load the most recent session (first in the sorted list)
        latest_session_id = updated_sessions[0][1]  # Get session_id from the first tuple
        history, loaded_session_id = load_chat_history(latest_session_id)
        logger.info(f"Loaded most recent session: {latest_session_id}")
        
        return gr.update(choices=updated_sessions, value=latest_session_id), "", history, ""
    
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        return gr.update(choices=get_chat_sessions(), value=""), "", [], f"Error deleting chat: {str(e)}"

def chat_function(message, history, session_id, input_type="text"):
    try:
        call_llm.current_session_id = session_id
        messages = [HumanMessage(content=message)]
        result = rag_agent.invoke({"messages": messages})
        ai_response = result['messages'][-1].content
        chat_entry = {
            "user_input": message,
            "ai_response": ai_response,
            "timestamp": datetime.utcnow(),
            "session_id": session_id,
            "input_type": input_type
        }
        collection.insert_one(chat_entry)
        if input_type == "voice":
            voice_entry = {
                "session_id": session_id,
                "transcript": message,
                "ai_response": ai_response,
                "timestamp": datetime.utcnow(),
                "audio_duration_seconds": None
            }
            voice_collection.insert_one(voice_entry)
        logger.info(f"Chat history saved: {chat_entry}")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ai_response})
        return history, ""
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, f"Error: {str(e)}"

def new_chat():
    new_session_id = str(uuid.uuid4())
    logger.info(f"Starting new chat with session_id: {new_session_id}")
    
    # Save an initial empty chat entry to MongoDB to make the session visible
    initial_chat_entry = {
        "user_input": "New chat started",
        "ai_response": "Chat initialized",
        "timestamp": datetime.utcnow(),
        "session_id": new_session_id,
        "input_type": "system"
    }
    try:
        collection.insert_one(initial_chat_entry)
        logger.info(f"Initial chat entry saved for session {new_session_id}")
    except Exception as e:
        logger.error(f"Error saving initial chat entry: {e}")
    
    # Refresh the chat sessions for the dropdown
    updated_sessions = get_chat_sessions()
    
    return "", [], new_session_id, gr.update(choices=updated_sessions, value=new_session_id), ""

def on_chat_selection_change(selected_value):
    if not selected_value or selected_value == "No previous chats" or selected_value == "Error loading chats":
        return [], selected_value, ""
    if isinstance(selected_value, tuple):
        session_id = selected_value[1]
    else:
        session_id = selected_value
    history, _ = load_chat_history(session_id)
    logger.info(f"Chat selection changed. Loading session: {session_id}")
    return history, session_id, ""

def handle_start_recording(history, session_id):
    try:
        transcript, error = voice_recognizer.start_recording()
        if error or transcript.startswith("Error"):
            return history, f"❌ {transcript}"
        return history, "🔴 Recording... Speak now!"
    except Exception as e:
        logger.error(f"Error starting voice recording: {e}")
        return history, f"❌ Error starting recording: {str(e)}"

def handle_stop_recording(history, session_id):
    try:
        transcript, error = voice_recognizer.stop_recording()
        if error or transcript.startswith("Error") or transcript.startswith("No speech"):
            return history, f"❌ {transcript}"
        history, error_msg = chat_function(transcript, history, session_id, "voice")
        if error_msg:
            return history, f"❌ {error_msg}"
        return history, "✅ Voice input processed successfully!"
    except Exception as e:
        logger.error(f"Error stopping voice recording: {e}")
        return history, f"❌ Error processing audio: {str(e)}"

def initialize_voice_system():
    success = voice_recognizer.initialize_models()
    if success:
        return "✅ Voice system initialized!"
    else:
        return "❌ Failed to initialize voice system"

with gr.Blocks(theme="soft", title="Voice-Enabled RAG Chatbot") as interface:
    with gr.Row():
        gr.Markdown("# 🎙️📊 Voice-Enabled RAG Chatbot")
    
    gr.Markdown("Ask questions about MapReduce or anything else using text or voice. Navigate between your previous chats using the sidebar.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### 💬 Chat History")
            chat_sessions = gr.Dropdown(
                label="Previous Chats",
                choices=get_chat_sessions(),
                value="",
                interactive=True,
                allow_custom_value=False
            )
            with gr.Row():
                delete_chat_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
            gr.Markdown("---")
            gr.Markdown("### 🎙️ Voice Input")
            voice_status = gr.Textbox(
                label="Voice System Status",
                value="Initializing voice system...",
                interactive=False,
                lines=6
            )
            with gr.Row():
                start_record_btn = gr.Button("🎤 Start Recording", variant="secondary", size="sm")
                stop_record_btn = gr.Button("🛑 Stop Recording", variant="stop", size="sm")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                show_share_button=True,
                avatar_images=["👤", "🤖"],
                type="messages"
            )
            with gr.Row():
                textbox = gr.Textbox(
                    placeholder="Ask me anything about MapReduce... (or use voice input)",
                    container=False,
                    scale=7,
                    lines=1
                )
                submit_btn = gr.Button("📤 Send", variant="primary", scale=1)
    
    state = gr.State(value=str(uuid.uuid4()))
    error_message = gr.Textbox(value="", visible=False)
    
    interface.load(
        fn=initialize_voice_system,
        outputs=[voice_status]
    )
    
    def send_message(message, history, session_id):
        if not message.strip():
            return history, "", ""
        updated_history, error_msg = chat_function(message, history, session_id, "text")
        return updated_history, "", error_msg
    
    submit_btn.click(
        fn=send_message,
        inputs=[textbox, chatbot, state],
        outputs=[chatbot, textbox, error_message]
    )
    
    textbox.submit(
        fn=send_message,
        inputs=[textbox, chatbot, state],
        outputs=[chatbot, textbox, error_message]
    )
    
    start_record_btn.click(
        fn=handle_start_recording,
        inputs=[chatbot, state],
        outputs=[chatbot, voice_status]
    )
    
    stop_record_btn.click(
        fn=handle_stop_recording,
        inputs=[chatbot, state],
        outputs=[chatbot, voice_status]
    )
    
    chat_sessions.change(
        fn=on_chat_selection_change,
        inputs=[chat_sessions],
        outputs=[chatbot, state, error_message]
    )
    
    delete_chat_btn.click(
        fn=delete_chat_session,
        inputs=[chat_sessions],
        outputs=[chat_sessions, textbox, chatbot, error_message]
    )
    
    new_chat_btn.click(
        fn=new_chat,
        inputs=[],
        outputs=[textbox, chatbot, state, chat_sessions, error_message]
    )
    
    def refresh_sessions():
        try:
            return gr.update(choices=get_chat_sessions())
        except Exception as e:
            logger.error(f"Error refreshing sessions: {e}")
            return gr.update(choices=[("Error loading chats", "")])

    
    refresh_btn.click(
        fn=refresh_sessions,
        outputs=[chat_sessions]
    )
    
    with gr.Row():
        gr.Examples(
            examples=[
                "What is MapReduce and how does it work?",
                "Explain the key components of MapReduce",
                "What are the advantages of using MapReduce?",
                "How does fault tolerance work in MapReduce?",
                "Describe the MapReduce programming model"
            ],
            inputs=[textbox],
            cache_examples=False
        )

def cleanup_resources():
    try:
        voice_recognizer.cleanup()
        if client:
            client.close()
        print("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        interface.launch()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cleanup_resources()