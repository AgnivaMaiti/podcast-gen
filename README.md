# StudySauce - PDF to Podcast Generator

Transform your PDF study materials into engaging podcast discussions using AI.

## 🚀 Local Development Setup

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Google Gemini API Key
- ElevenLabs API Key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/studysauce-podcast-generator.git
   cd studysauce-podcast-generator
   ```

2. **Set up environment variables**
   Create a `.env` file in the project root with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

3. **Install dependencies**
   Run the setup script:
   ```bash
   setup.bat  # On Windows
   ```

4. **Run the application**
   ```bash
   # Activate virtual environment (Windows)
   venv\Scripts\activate
   
   # Run the Flask app
   python app.py
   ```

5. **Access the application**
   Open your browser and visit: http://localhost:5000

## 🎯 Features
- Convert PDF study materials into podcast discussions
- Customize number of hosts and guests
- Adjust podcast length (3-15 minutes)
- Preview and download generated podcasts

## 📝 Project Structure
- `app.py` - Main Flask application
- `templates/` - HTML templates
- `uploads/` - Temporary storage for uploaded PDFs
- `output/` - Generated podcast files