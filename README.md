# Gemini 2.0 Flash AI Chatbot with Gradio Frontend

A modern chatbot interface powered by Google's latest Gemini 2.0 Flash AI model with a user-friendly Gradio web interface.

## Features

- 🤖 **Powered by Gemini 2.0 Flash**: Uses the latest Gemini 2.0 Flash model for fast and intelligent responses
- 💬 **Interactive Chat Interface**: Clean and intuitive web-based chat interface
- 📝 **Context Awareness**: Maintains conversation history for better responses
- 🎨 **Modern UI**: Beautiful Gradio interface with responsive design
- 🔄 **Real-time Responses**: Instant AI responses with typing indicators
- 🧹 **Clear Chat**: Option to clear conversation history

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key (already configured in .env file)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the chatbot:
```bash
python ../clineTest.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

3. Start chatting with the Gemini AI!

## Configuration

The chatbot uses the Gemini API key from the `.env` file. The key is already configured as:
- `GEMINI_API_KEY`: Your Google Gemini API key

## Interface Features

### Main Chat Area
- **Chat History**: Displays the conversation between you and the AI
- **Message Input**: Type your messages here
- **Send Button**: Click to send your message
- **Clear Chat**: Reset the conversation history

### Sidebar Information
- **Tips**: Helpful suggestions for interacting with the AI
- **Features**: Overview of chatbot capabilities

## Example Interactions

- Ask questions: "What is machine learning?"
- Get explanations: "Explain quantum computing in simple terms"
- Coding help: "How do I create a Python function?"
- Creative tasks: "Write a short story about space exploration"
- Recommendations: "Suggest some good books on AI"

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check if your Gemini API key is valid
   - Ensure you have internet connectivity
   - Verify the API key in the .env file

2. **Module Not Found**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

3. **Port Already in Use**
   - The default port is 7860. If it's busy, the app will try to find another available port

## Technical Details

### Architecture
- **Backend**: Python with Google Generative AI library
- **Frontend**: Gradio web interface
- **Model**: Google Gemini 2.0 Flash
- **Configuration**: Environment variables via python-dotenv

### File Structure
```
├── clineTest.py          # Main chatbot application
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (API keys)
└── README.md           # This file
```

## Customization

You can customize the chatbot by modifying `clineTest.py`:

- Change the Gemini model (e.g., to 'gemini-pro-vision' for image support)
- Modify the UI theme and layout
- Add custom system prompts
- Implement conversation memory persistence
- Add file upload capabilities

## License

This project is open source and available under the MIT License.
