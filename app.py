from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import re
import time
import random
import io
import tempfile
from datetime import datetime
import fitz 
from pydub import AudioSegment
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

app = Flask(__name__)
CORS(app)

# Use /tmp for serverless environments
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL') else 'uploads'
OUTPUT_FOLDER = '/tmp/output' if os.environ.get('VERCEL') else 'output'
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

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
    response = model.generate_content(prompt)
    text = response.text or ""
    return text.strip()

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
            raise e2

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
        
        final_audio = AudioSegment.silent(duration=1000)
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
                    final_audio += audio_seg + AudioSegment.silent(duration=500)
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
        output_filename = f"StudySauce_Podcast_{timestamp}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        final_audio.export(output_path, format="mp3")
        
        processing_status[task_id]['status'] = 'Complete!'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['download_url'] = f'/download/{output_filename}'
        processing_status[task_id]['filename'] = output_filename
        
    except Exception as e:
        processing_status[task_id]['status'] = f'Error: {str(e)}'
        processing_status[task_id]['progress'] = 0
        processing_status[task_id]['error'] = True

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>StudySauce - PDF to Podcast</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
            .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
            .hidden { display: none; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #45a049; }
            button:disabled { background: #cccccc; cursor: not-allowed; }
            .customization-panel { background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100%; max-width: 200px; }
            .hosts-selection { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 10px; }
            .host-option { display: flex; align-items: center; }
            .host-option input { width: auto; margin-right: 8px; }
            .slider-container { display: flex; align-items: center; gap: 10px; }
            .slider { width: 200px; }
            .slider-value { font-weight: bold; color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>🎧 StudySauce - PDF to Podcast Generator</h1>
        <p>Upload a PDF and we'll create an engaging podcast discussion about its contents!</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept=".pdf" style="display: none;" onchange="showCustomization()">
            <p>Click here to select a PDF file</p>
            <p style="font-size: 0.9em; color: #666;">Maximum file size: 16MB</p>
        </div>
        
        <div id="customizationPanel" class="customization-panel hidden">
            <h3>🎛️ Customize Your Podcast</h3>
            
            <div class="form-group">
                <label for="numHosts">Number of Hosts (1-4):</label>
                <select id="numHosts" onchange="updateHostSelection()">
                    <option value="1">1 Host</option>
                    <option value="2" selected>2 Hosts</option>
                    <option value="3">3 Hosts</option>
                    <option value="4">4 Hosts</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="numGuests">Number of Guests (0-3):</label>
                <select id="numGuests" onchange="updateHostSelection()">
                    <option value="0">No Guests</option>
                    <option value="1" selected>1 Guest</option>
                    <option value="2">2 Guests</option>
                    <option value="3">3 Guests</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="podcastLength">Podcast Length:</label>
                <div class="slider-container">
                    <input type="range" id="podcastLength" class="slider" min="3" max="15" value="10" oninput="updateLengthDisplay()">
                    <span class="slider-value" id="lengthValue">10 minutes</span>
                </div>
            </div>
            
            <div class="form-group">
                <label>Choose Your Hosts (optional):</label>
                <div class="hosts-selection" id="hostsSelection">
                    <!-- Host checkboxes will be populated by JavaScript -->
                </div>
            </div>
            
            <button onclick="uploadFile()" id="generateBtn">Generate Podcast 🎙️</button>
        </div>
        
        <div id="progressSection" class="hidden">
            <h3>Processing...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
            <p id="statusText">Initializing...</p>
        </div>
        
        <div id="downloadSection" class="hidden">
            <h3>✅ Podcast Ready!</h3>
            <button id="downloadBtn" onclick="downloadPodcast()">Download Podcast</button>
            <audio id="audioPlayer" controls style="width: 100%; margin-top: 20px;"></audio>
        </div>

        <script>
            let currentTaskId = null;
            const availableHosts = ["Alex", "Taylor", "Jordan", "Casey", "Morgan", "Riley", "Dakota", "Harper", "Quinn", "Reese"];
            
            function showCustomization() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                if (file.size > 16 * 1024 * 1024) {
                    alert('File size must be less than 16MB');
                    return;
                }
                
                document.getElementById('customizationPanel').classList.remove('hidden');
                updateHostSelection();
            }
            
            function updateHostSelection() {
                const numHosts = parseInt(document.getElementById('numHosts').value);
                const numGuests = parseInt(document.getElementById('numGuests').value);
                const totalSpeakers = numHosts + numGuests;
                const hostsContainer = document.getElementById('hostsSelection');
                
                hostsContainer.innerHTML = `
                    <p style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                        Select up to ${totalSpeakers} names (optional). If you select fewer, we'll randomly choose the rest.
                    </p>
                `;
                
                availableHosts.forEach(host => {
                    const hostDiv = document.createElement('div');
                    hostDiv.className = 'host-option';
                    hostDiv.innerHTML = `
                        <input type="checkbox" id="host_${host}" name="selectedHosts" value="${host}" 
                               onchange="checkSelectionLimit(${totalSpeakers})">
                        <label for="host_${host}">${host}</label>
                    `;
                    hostsContainer.appendChild(hostDiv);
                });
            }
            
            function checkSelectionLimit(maxSelections) {
                const selectedBoxes = document.querySelectorAll('input[name="selectedHosts"]:checked');
                const allBoxes = document.querySelectorAll('input[name="selectedHosts"]');
                
                if (selectedBoxes.length >= maxSelections) {
                    // Disable unchecked boxes
                    allBoxes.forEach(box => {
                        if (!box.checked) {
                            box.disabled = true;
                        }
                    });
                } else {
                    // Enable all boxes
                    allBoxes.forEach(box => {
                        box.disabled = false;
                    });
                }
            }
            
            function updateLengthDisplay() {
                const length = document.getElementById('podcastLength').value;
                document.getElementById('lengthValue').textContent = length + ' minutes';
            }
            
            function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                const formData = new FormData();
                formData.append('pdf', file);
                formData.append('num_hosts', document.getElementById('numHosts').value);
                formData.append('num_guests', document.getElementById('numGuests').value);
                formData.append('podcast_length', document.getElementById('podcastLength').value);
                
                // Add selected hosts
                const selectedHosts = Array.from(document.querySelectorAll('input[name="selectedHosts"]:checked'))
                    .map(cb => cb.value);
                selectedHosts.forEach(host => {
                    formData.append('selected_hosts[]', host);
                });
                
                document.getElementById('customizationPanel').classList.add('hidden');
                document.getElementById('progressSection').classList.remove('hidden');
                document.getElementById('downloadSection').classList.add('hidden');
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.task_id) {
                        currentTaskId = data.task_id;
                        checkStatus();
                    } else {
                        alert('Upload failed: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Upload failed: ' + error);
                });
            }
            
            function checkStatus() {
                if (!currentTaskId) return;
                
                fetch(`/status/${currentTaskId}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progressFill').style.width = data.progress + '%';
                    document.getElementById('statusText').textContent = data.status;
                    
                    if (data.progress >= 100 && data.download_url) {
                        document.getElementById('downloadSection').classList.remove('hidden');
                        document.getElementById('audioPlayer').src = data.download_url;
                        
                        // Store filename for direct download
                        if (data.filename) {
                            document.getElementById('downloadBtn').onclick = function() {
                                window.location.href = `/download/${data.filename}`;
                            };
                        }
                    } else if (!data.error) {
                        setTimeout(checkStatus, 2000);
                    } else {
                        alert('Processing failed: ' + data.status);
                    }
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                    setTimeout(checkStatus, 5000);
                });
            }
            
            function downloadPodcast() {
                if (currentTaskId) {
                    // Direct download using the task_id
                    window.location.href = `/download/${currentTaskId}`;
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        num_hosts = int(request.form.get('num_hosts', 2))
        num_guests = int(request.form.get('num_guests', 1))
        podcast_length = int(request.form.get('podcast_length', 10))  # in minutes
        selected_hosts = request.form.getlist('selected_hosts[]')
        
        if num_hosts < 1 or num_hosts > 4:
            return jsonify({'error': 'Number of hosts must be between 1 and 4'}), 400
        if num_guests < 0 or num_guests > 3:
            return jsonify({'error': 'Number of guests must be between 0 and 3'}), 400
        if podcast_length < 3 or podcast_length > 15:
            return jsonify({'error': 'Podcast length must be between 3 and 15 minutes'}), 400
        
        task_id = f"task_{timestamp}_{random.randint(1000, 9999)}"
        
        processing_status[task_id] = {
            'status': 'Starting...',
            'progress': 0,
            'error': False,
            'num_hosts': num_hosts,
            'num_guests': num_guests,
            'podcast_length': podcast_length,
            'selected_hosts': selected_hosts
        }
        
        thread = threading.Thread(target=process_podcast_creation, args=(filepath, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/download/<identifier>')
def download_file(identifier):
    try:
        if identifier in processing_status and 'filename' in processing_status[identifier]:
            filename = processing_status[identifier]['filename']
        else:
            filename = identifier
        
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            filepath, 
            as_attachment=True, 
            download_name=filename,
            mimetype='audio/mpeg'
        )
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

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