from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import re
import time
import random
import io
import tempfile
from datetime import datetime
import fitz 
import sys
import platform
import subprocess
import shutil
import pydub
import warnings

# Suppress FFmpeg warnings
warnings.filterwarnings("ignore", message="Couldn't find ffprobe or avprobe")

# Try to import AudioSegment with FFmpeg fallback
try:
    AudioSegment = pydub.AudioSegment
    # Test FFmpeg availability
    test_audio = AudioSegment.silent(duration=100)
    FFMPEG_AVAILABLE = True
except Exception as e:
    print(f"FFmpeg not available, using basic audio processing: {str(e)}")
    FFMPEG_AVAILABLE = False
    
    # Define a simplified AudioSegment fallback
    class SimpleAudioSegment:
        def __init__(self, data=None, *args, **kwargs):
            self.data = data or b''
            self.frame_rate = 44100
            self.channels = 2
            self.sample_width = 2
            
        @classmethod
        def from_file(cls, file, format=None):
            with open(file, 'rb') as f:
                return cls(f.read())
                
        def export(self, output_path, format='wav'):
            with open(output_path, 'wb') as f:
                f.write(self.data)
                
        def __add__(self, other):
            if isinstance(other, SimpleAudioSegment):
                return SimpleAudioSegment(self.data + other.data)
            return self
            
        @classmethod
        def silent(cls, duration=1000, *args, **kwargs):
            # Return a silent audio segment of the specified duration in ms
            return cls(b'\x00' * (duration * 44))  # Rough approximation for 44.1kHz stereo
    
    AudioSegment = SimpleAudioSegment
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream, save
from google.api_core.exceptions import BadRequest
import threading
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

# Check if running in Vercel
IS_VERCEL = os.environ.get('VERCEL') == '1'

app = Flask(__name__)
CORS(app)

# Configure FFmpeg path if on Windows and not in Vercel
if platform.system() == 'Windows' and not IS_VERCEL:
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Use /tmp for serverless environments
UPLOAD_FOLDER = '/tmp/uploads' if IS_VERCEL else 'uploads'
OUTPUT_FOLDER = '/tmp/output' if IS_VERCEL else 'output'
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GEMINI_API_KEY or not ELEVENLABS_API_KEY:
    raise ValueError("API keys not found. Please check your .env file")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

NUM_HOSTS = 2
NUM_GUESTS = 1
NAME_POOL = [
    "Alex", "Taylor", "Jordan", "Casey", "Morgan",
    "Riley", "Dakota", "Harper", "Quinn", "Reese"
]
MIN_SPLIT_SIZE = 1000
AVERAGE_WPM = 150.0

processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        text = response.text or ""
        return text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        raise

def split_on_sentence_boundary(text: str) -> tuple[str, str]:
    mid = len(text) // 2
    idx = text.rfind(". ", 0, mid)
    if idx == -1:
        return text[:mid].strip(), text[mid:].strip()
    else:
        return text[: idx + 1].strip(), text[idx + 1 :].strip()

def recursive_summarize(text: str) -> str:
    prompt = (
        "Summarize the following academic content in a concise, "
        "concept-focused manner. Preserve key ideas:\n\n"
        f"{text}"
    )
    try:
        return call_gemini(prompt)
    except BadRequest as e:
        if len(text) < MIN_SPLIT_SIZE:
            truncated = text[: min(len(text), MIN_SPLIT_SIZE)]
            try:
                return call_gemini(
                    "Summarize this small snippet concisely:\n\n" + truncated
                )
            except Exception:
                return truncated[:500] + "..."
        left, right = split_on_sentence_boundary(text)
        left_sum = recursive_summarize(left)
        right_sum = recursive_summarize(right)
        return f"{left_sum} {right_sum}"

def process_pdf_to_summary(pdf_path: str, task_id: str) -> str:
    processing_status[task_id]['status'] = 'Processing PDF...'
    processing_status[task_id]['progress'] = 10
    
    with fitz.open(pdf_path) as doc:
        all_summaries = []
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text").strip()
            page_text = re.sub(r"\s+", " ", raw_text)
            
            if len(page_text) < 50:
                continue
            
            processing_status[task_id]['status'] = f'Summarizing page {page_num}/{total_pages}...'
            processing_status[task_id]['progress'] = 10 + (page_num / total_pages) * 30
            
            try:
                page_summary = recursive_summarize(page_text)
                all_summaries.append(page_summary)
                time.sleep(1)
            except Exception as e:
                fallback = page_text[:500] + "..."
                all_summaries.append(fallback)
        
        if not all_summaries:
            raise RuntimeError("No valid pages found in PDF.")
        
        concatenated = " ".join(all_summaries)
        final_summary = concatenated[:2000]
        return final_summary

def pick_random_names(num_total, pool, selected_names=None):
    if selected_names and len(selected_names) > 0:
        available_names = [name for name in pool if name not in selected_names]
        if len(selected_names) >= num_total:
            return selected_names[:num_total]
        else:
            remaining_needed = num_total - len(selected_names)
            additional_names = random.sample(available_names, k=min(remaining_needed, len(available_names)))
            return selected_names + additional_names
    else:
        return random.sample(pool, k=num_total)

def build_voice_settings(speakers):
    settings = {}
    try:
        voices_response = eleven_client.voices.get_all()
        
        if hasattr(voices_response, 'voices'):
            all_voices = voices_response.voices
        else:
            all_voices = voices_response
        
        filtered = []
        for voice in all_voices:
            if hasattr(voice, 'voice_id') and hasattr(voice, 'name'):
                if voice.voice_id and voice.name:
                    filtered.append(voice)
            elif isinstance(voice, dict):
                if voice.get('voice_id') and voice.get('name'):
                    filtered.append(voice)
        
        if not filtered:
            fallback_voices = [
                {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
                {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
                {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
                {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"}
            ]
            filtered = fallback_voices
        
        for name in speakers:
            choice = random.choice(filtered)
            if isinstance(choice, dict):
                settings[name.lower()] = {
                    "voice_id": choice['voice_id'],
                    "voice_name": choice['name']
                }
            else:
                settings[name.lower()] = {
                    "voice_id": choice.voice_id,
                    "voice_name": choice.name
                }
        
        return settings
    
    except Exception as e:
        fallback_voices = [
            {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
            {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
            {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
            {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
            {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"}
        ]
        
        settings = {}
        for i, name in enumerate(speakers):
            voice = fallback_voices[i % len(fallback_voices)]
            settings[name.lower()] = {
                "voice_id": voice["voice_id"],
                "voice_name": voice["name"]
            }
        
        return settings

def generate_podcast_script(summary: str, speakers: list[str], task_id: str) -> str:
    processing_status[task_id]['status'] = 'Generating podcast script...'
    processing_status[task_id]['progress'] = 45
    
    num_hosts = processing_status[task_id]['num_hosts']
    num_guests = processing_status[task_id]['num_guests']
    podcast_length = processing_status[task_id]['podcast_length']
    
    hosts = speakers[:num_hosts]
    guests = speakers[num_hosts:num_hosts + num_guests] if num_guests > 0 else []
    
    intro_speakers = ", ".join(hosts)
    if guests:
        intro_speakers += " with guest" + ("s" if len(guests) > 1 else "") + " " + ", ".join(guests)
    
    if podcast_length <= 5:
        length_instruction = "Keep the conversation concise and focused, suitable for a 3-5 minute podcast."
    elif podcast_length <= 10:
        length_instruction = f"Create a {podcast_length}-minute conversation with good depth but not too lengthy."
    else:
        length_instruction = f"Create a comprehensive {podcast_length}-minute discussion with detailed exploration of topics."
    
    prompt = f"""
Generate a conversational podcast script titled 'StudySauce' with {intro_speakers}.
They will discuss the following research summary in a natural, engaging way.

Research Summary:
{summary}

Guidelines:
- Start with a casual intro by one of the hosts (e.g. "Hey everyone, welcome to StudySauce...")
- Alternate dialogue lines between speakers (e.g. "Alex:" / "Taylor:" / "Morgan:")
- Keep it free-flowing—no visible headers like "Introduction" or "Conclusion."
- Include insightful commentary, light humor, and deeper reflections.
- {length_instruction}
- Target approximately {podcast_length * 150} words total (average 150 words per minute).
"""
    resp = model.generate_content(prompt)
    script_text = resp.text.strip()
    return script_text

def clean_line(line: str) -> str:
    return re.sub(r"\*+", "", line).strip()

def is_dialogue_line(line: str, speakers_lower: set[str]) -> bool:
    lower = line.lower().lstrip()
    return any(lower.startswith(f"{speaker}:") for speaker in speakers_lower)

def get_voice_settings_for_line(line: str, voice_settings: dict) -> dict:
    lower = line.lower()
    for name, cfg in voice_settings.items():
        if lower.startswith(f"{name}:"):
            return cfg
    return {"voice_id": None, "voice_name": None}

def strip_speaker_label(line: str) -> str:
    return re.sub(r"^[A-Za-z]+:\s*", "", line)

def text_to_audio_elevenlabs(text: str, voice_id: str) -> AudioSegment:
    try:
        audio_generator = eleven_client.generate(
            text=text,
            voice=voice_id,
            model="eleven_flash_v2_5"
        )
        
        audio_data = b"".join(audio_generator)
        return AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        
    except Exception as e:
        try:
            stream = eleven_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.3,
                    "similarity_boost": 0.6,
                    "style": 0.1,
                    "use_speaker_boost": False
                }
            )
            data = b"".join(stream)
            return AudioSegment.from_file(io.BytesIO(data), format="mp3")
        except Exception as e2:
            print(f"Audio generation failed: {e2}")
            raise Exception(f"Failed to generate audio: {str(e2)}")

def process_podcast_creation(pdf_path: str, task_id: str):
    try:
        num_hosts = processing_status[task_id]['num_hosts']
        num_guests = processing_status[task_id]['num_guests']
        selected_hosts = processing_status[task_id]['selected_hosts']
        
        summary = process_pdf_to_summary(pdf_path, task_id)
        
        total_speakers = num_hosts + num_guests
        speakers = pick_random_names(total_speakers, NAME_POOL, selected_hosts)
        voice_settings = build_voice_settings(speakers)
        
        script = generate_podcast_script(summary, speakers, task_id)
        
        processing_status[task_id]['status'] = 'Converting to audio...'
        processing_status[task_id]['progress'] = 60
        
        # Initialize audio based on FFmpeg availability
        if FFMPEG_AVAILABLE:
            final_audio = AudioSegment.silent(duration=1000)  # Start with 1s of silence
            audio_format = "mp3"
        else:
            # Use WAV format if FFmpeg is not available
            final_audio = AudioSegment.silent(duration=1000)
            audio_format = "wav"
            
        speakers_lower = set(name.lower() for name in speakers)
        
        script_lines = [clean_line(line) for line in script.split("\n") if clean_line(line)]
        total_lines = len([line for line in script_lines if is_dialogue_line(line, speakers_lower)])
        processed_lines = 0
        
        for line in script_lines:
            if not line:
                continue
            
            if is_dialogue_line(line, speakers_lower):
                cfg = get_voice_settings_for_line(line, voice_settings)
                voice_id = cfg["voice_id"]
                content = strip_speaker_label(line)
                
                if not voice_id:
                    continue
                
                try:
                    audio_seg = text_to_audio_elevenlabs(text=content, voice_id=voice_id)
                    
                    if FFMPEG_AVAILABLE:
                        final_audio += audio_seg + AudioSegment.silent(duration=500)
                    else:
                        # Simple concatenation for non-FFmpeg mode
                        if isinstance(audio_seg, bytes):
                            final_audio.data += audio_seg
                        else:
                            final_audio.data += audio_seg.data if hasattr(audio_seg, 'data') else b''
                    
                    processed_lines += 1
                    progress = 60 + (processed_lines / total_lines) * 35
                    processing_status[task_id]['progress'] = min(progress, 95)
                    processing_status[task_id]['status'] = f'Processing audio ({processed_lines}/{total_lines})...'
                    
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        processing_status[task_id]['status'] = 'Finalizing podcast...'
        processing_status[task_id]['progress'] = 95
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"StudySauce_Podcast_{timestamp}.{audio_format}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Export based on available capabilities
        if FFMPEG_AVAILABLE:
            final_audio.export(output_path, format=audio_format)
        else:
            # For non-FFmpeg mode, just write the raw data
            with open(output_path, 'wb') as f:
                if hasattr(final_audio, 'data'):
                    f.write(final_audio.data)
                else:
                    f.write(final_audio)
        
        processing_status[task_id].update({
            'status': 'Complete!',
            'progress': 100,
            'filename': output_filename,
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        error_msg = f"Error in process_podcast_creation: {str(e)}"
        print(f"\n!!! ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        
        if task_id in processing_status:
            processing_status[task_id].update({
                'status': f'Error: {str(e)}',
                'progress': 0,
                'error': True,
                'error_details': str(e),
                'traceback': traceback.format_exc(),
                'completed_at': datetime.now().isoformat()
            })
        raise Exception(error_msg)

# In non-Vercel environments, also save the MP3 file locally
if not IS_VERCEL:
    def save_mp3_locally(final_audio, output_path, task_id, output_filename):
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as MP3
            final_audio.export(output_path, format="mp3")
            
            # Also save WAV data for download endpoint
            with io.BytesIO() as wav_buffer:
                final_audio.export(wav_buffer, format="wav")
                wav_data = wav_buffer.getvalue()
            
            # Update status
            processing_status[task_id].update({
                'status': 'Complete!',
                'progress': 100,
                'filename': output_filename,
                'wav_data': wav_data.hex(),
                'file_size': f"{os.path.getsize(output_path) / (1024 * 1024):.1f} MB",
                'duration_seconds': len(final_audio) / 1000,
                'completed_at': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"Error saving MP3: {str(e)}")
            return False
    
    def save_wav_locally(final_audio, output_path, task_id, output_filename):
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as WAV
            final_audio.export(output_path, format="wav")
            
            # Update status for WAV
            processing_status[task_id].update({
                'status': 'Complete! (WAV format)',
                'progress': 100,
                'filename': output_filename,
                'file_size': f"{os.path.getsize(output_path) / (1024 * 1024):.1f} MB",
                'duration_seconds': len(final_audio) / 1000,
                'completed_at': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"Error saving WAV: {str(e)}")
            return False
else:
    # For Vercel environment, use these placeholder functions
    def save_mp3_locally(*args, **kwargs):
        print("Running in Vercel environment - MP3 save not available")
        return False
    
    def save_wav_locally(*args, **kwargs):
        print("Running in Vercel environment - WAV save not available")
        return False

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/main')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("\n=== Upload Request ===")
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
        
        if 'pdf' not in request.files:
            error_msg = 'No file uploaded'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg, 'details': 'No file part in the request'}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            error_msg = 'No file selected'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        if not file or not allowed_file(file.filename):
            error_msg = 'Invalid file type. Please upload a PDF.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg, 'details': f'File type: {file.content_type}'}), 400
        
        try:
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Get form data with error handling
            try:
                num_hosts = int(request.form.get('num_hosts', 2))
                num_guests = int(request.form.get('num_guests', 1))
                podcast_length = int(request.form.get('podcast_length', 10))
                selected_hosts = request.form.getlist('selected_hosts[]')
                
                print(f"Form data - Hosts: {num_hosts}, Guests: {num_guests}, Length: {podcast_length}min")
                print(f"Selected hosts: {selected_hosts}")
                
                # Validate inputs
                if num_hosts < 1 or num_hosts > 4:
                    raise ValueError('Number of hosts must be between 1 and 4')
                if num_guests < 0 or num_guests > 3:
                    raise ValueError('Number of guests must be between 0 and 3')
                if podcast_length < 3 or podcast_length > 15:
                    raise ValueError('Podcast length must be between 3 and 15 minutes')
                
            except ValueError as ve:
                error_msg = str(ve)
                print(f"Validation error: {error_msg}")
                return jsonify({'error': error_msg}), 400
            
            # Create task
            task_id = f"task_{timestamp}_{random.randint(1000, 9999)}"
            
            processing_status[task_id] = {
                'status': 'Starting...',
                'progress': 0,
                'error': False,
                'num_hosts': num_hosts,
                'num_guests': num_guests,
                'podcast_length': podcast_length,
                'selected_hosts': selected_hosts,
                'start_time': time.time()
            }
            
            print(f"Starting processing task: {task_id}")
            
            # Start processing in a separate thread
            thread = threading.Thread(
                target=process_podcast_creation, 
                args=(filepath, task_id),
                daemon=True
            )
            thread.start()
            
            return jsonify({
                'task_id': task_id,
                'message': 'Processing started',
                'status_url': f'/status/{task_id}'
            })
            
        except Exception as e:
            error_msg = f'Error processing file: {str(e)}'
            print(f"Processing error: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': 'Failed to process file',
                'details': str(e),
                'traceback': traceback.format_exc()
            }), 500
            
    except Exception as e:
        error_msg = f'Unexpected error: {str(e)}'
        print(f"Unexpected error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    try:
        if task_id not in processing_status:
            return jsonify({'error': 'Task not found'}), 404
            
        status_data = processing_status[task_id].copy()
        
        # Ensure we have all required fields
        if 'status' not in status_data:
            status_data['status'] = 'Processing...'
        if 'progress' not in status_data:
            status_data['progress'] = 0
            
        # If the task is complete, include download URL if available
        if status_data.get('status') == 'Complete!' and 'filename' in status_data:
            status_data['download_url'] = f'/download/{status_data["filename"]}'
            
        return jsonify(status_data)
        
    except Exception as e:
        print(f"Error in get_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<identifier>', methods=['GET'])
def download_file(identifier):
    try:
        # Check if this is a task ID with associated file
        if identifier in processing_status and 'filename' in processing_status[identifier]:
            filename = processing_status[identifier]['filename']
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            # If we have WAV data in memory, use that instead of the file
            if 'wav_data' in processing_status[identifier]:
                print(f"Serving WAV data from memory for task {identifier}")
                wav_hex = processing_status[identifier]['wav_data']  # already hex-encoded
                # Ensure a .wav filename for client download
                wav_filename = filename.replace('.mp3', '.wav') if filename.lower().endswith('.mp3') else filename
        
        # Look for the file with the given identifier in the filename
        for file in os.listdir(output_dir):
            if identifier in file:
                filename = file
                break
                
        if not filename:
            return jsonify({'error': 'File not found'}), 404
            
        filepath = os.path.join(output_dir, filename)
        
        # Determine MIME type based on file extension
        if filename.lower().endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.lower().endswith('.wav'):
            mimetype = 'audio/wav'
        else:
            mimetype = 'application/octet-stream'  # Fallback MIME type
        
        # If FFmpeg is not available and the file is MP3, try to convert from WAV
        if not FFMPEG_AVAILABLE and filename.lower().endswith('.mp3'):
            try:
                # Look for WAV version of the file
                wav_filename = os.path.splitext(filename)[0] + '.wav'
                wav_path = os.path.join(output_dir, wav_filename)
                
                if os.path.exists(wav_path):
                    # Serve the WAV version instead
                    return send_file(
                        wav_path,
                        as_attachment=True,
                        download_name=wav_filename,
                        mimetype='audio/wav'
                    )
            except Exception as e:
                print(f"Error serving WAV fallback: {str(e)}")
                # Continue with the original file if WAV conversion fails
        
        # Return the file for download
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    print("🎧 StudySauce Backend Starting...")
    print("📋 Make sure to set your API keys as environment variables:")
    print("   export GEMINI_API_KEY='your_gemini_key'")
    print("   export ELEVENLABS_API_KEY='your_elevenlabs_key'")
    print("\n🌐 Starting Flask server...")
    app.run(debug=True, port=5000)