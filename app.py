from flask import Flask, request, jsonify, send_file, url_for,Response, make_response
import re
from flask_cors import CORS
from functools import wraps
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from utils.image_utils import create_video_from_image, is_image_file
from utils.image_config import DEFAULT_VIDEO_DURATION, DEFAULT_VIDEO_FPS
import secrets
import logging
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Load environment variables at startup
load_dotenv()

# Configuration
class Config:
    # Use getenv with a generated fallback
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:  # If no API key in env, generate one
        API_KEY = secrets.token_hex(32)
        print(f"[WARNING] No API key found in environment")
        print(f"Generated new API key: {API_KEY}")
        print("Please save this key and add it to your .env file")
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    TEMP_FOLDER = 'temp'
    MAX_CONTENT_LENGTH = 99 * 1024 * 1024  # 99MB max file size
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm'}
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg'}
    
    # CORS Settings
    CORS_HEADERS = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, X-API-Key'
    }

# Create necessary folders
for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, Config.TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename='api_access.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API Key Authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        client_ip = request.remote_addr
        
        if not api_key:
            logging.warning(f'Missing API key from {client_ip}')
            return jsonify({'error': 'Missing API key'}), 401
            
        if not api_key == Config.API_KEY:
            logging.warning(f'Invalid API key attempt from {client_ip}')
            return jsonify({'error': 'Invalid API key'}), 401
            
        logging.info(f'Successful API access from {client_ip}')
        return f(*args, **kwargs)
    return decorated_function

# Add a route to generate new API keys (admin only)
def require_admin_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_key = request.headers.get('X-Admin-Key')
        if not admin_key or admin_key != os.getenv('ADMIN_KEY'):
            return jsonify({'error': 'Invalid admin credentials'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Utility Functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, prefix):
    """Save uploaded file with unique name"""
    unique_id = str(uuid.uuid4())
    filename = secure_filename(f"{prefix}_{unique_id}_{file.filename}")
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)
    return {'id': unique_id, 'path': filepath}

def convert_audio_to_wav(input_path):
    """Convert audio to WAV format if needed"""
    if input_path.lower().endswith('.wav'):
        return input_path
    
    output_path = os.path.splitext(input_path)[0] + '.wav'
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format='wav')
        return output_path
    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}")

def run_inference(face_path, audio_path, output_path):
    """Run the inference script"""
    cmd = [
        'python', 'inference.py',
        '--face', face_path,
        '--audio', audio_path,
        '--outfile', output_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        raise Exception(f"Inference failed: {stderr.decode('utf-8')}")
    
    return output_path

def download_from_url(url, temp_dir, allowed_extensions):
    """Download file from URL and validate"""
    try:
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        if not ext or ext[1:] not in allowed_extensions:
            return None, f'Invalid file type {ext}. Allowed types: {allowed_extensions}'
        
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(temp_dir, filename)
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None, f'Failed to download from URL: {response.status_code}'
            
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        return filepath, None
        
    except Exception as e:
        return None, f'URL download failed: {str(e)}'

def get_full_url(path):
    """Generate full URL including server address"""
    # Get server's base URL
    base_url = request.host_url.rstrip('/')  # e.g. "http://204.12.229.26:5000"
    return f"{base_url}{path}"

# API Routes
@app.route('/api/store/face', methods=['POST'])
@require_api_key
def store_face():
    """Store face image/video from file upload or URL"""
    try:
        # Handle URL input
        if 'url' in request.form:
            url = request.form['url']
            file_type = request.form.get('type', 'image')  # 'image' or 'video'
            allowed_extensions = (Config.ALLOWED_IMAGE_EXTENSIONS 
                               if file_type == 'image' 
                               else Config.ALLOWED_VIDEO_EXTENSIONS)
            
            filepath, error = download_from_url(url, Config.UPLOAD_FOLDER, allowed_extensions)
            if error:
                return jsonify({'error': f'URL error: {error}'}), 400
                
            file_id = str(uuid.uuid4())
            return jsonify({
                'status': 'success',
                'message': 'Face file stored successfully',
                'data': {
                    'file_id': file_id,
                    'filepath': filepath,
                    'type': file_type
                }
            }), 200
            
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_type = request.form.get('type', 'image')
        
        if file_type == 'image' and not allowed_file(file.filename, Config.ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image format'}), 400
        elif file_type == 'video' and not allowed_file(file.filename, Config.ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': 'Invalid video format'}), 400
        
        result = save_uploaded_file(file, f'face_{file_type}')
        return jsonify({
            'status': 'success',
            'message': 'Face file uploaded successfully',
            'data': {
                'file_id': result['id'],
                'filepath': result['path'],
                'type': file_type
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/store/audio', methods=['POST'])
@require_api_key
def store_audio():
    """Store audio from file upload or URL"""
    try:
        # Handle URL input
        if 'url' in request.form:
            url = request.form['url']
            filepath, error = download_from_url(url, Config.UPLOAD_FOLDER, Config.ALLOWED_AUDIO_EXTENSIONS)
            if error:
                return jsonify({'error': f'URL error: {error}'}), 400
                
            # Convert to WAV if needed
            if not filepath.lower().endswith('.wav'):
                wav_path = convert_audio_to_wav(filepath)
                os.remove(filepath)  # Remove original file
                filepath = wav_path
                
            file_id = str(uuid.uuid4())
            return jsonify({
                'status': 'success',
                'message': 'Audio file stored successfully',
                'data': {
                    'file_id': file_id,
                    'filepath': filepath
                }
            }), 200
            
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename, Config.ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': 'Invalid audio format'}), 400
            
        result = save_uploaded_file(file, 'audio')
        
        # Convert to WAV if needed
        if not file.filename.lower().endswith('.wav'):
            result['path'] = convert_audio_to_wav(result['path'])
            
        return jsonify({
            'status': 'success',
            'message': 'Audio file stored successfully',
            'data': {
                'file_id': result['id'],
                'filepath': result['path']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/process', methods=['POST'])
@require_api_key
def process():
    """Process files - handles both local uploads and URLs"""
    face_path = None
    audio_path = None
    temp_dir = "temp"
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Handle face/image input
        if 'face' in request.files:
            face_file = request.files['face']
            if face_file and allowed_file(face_file.filename):
                face_filename = secure_filename(face_file.filename)
                face_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{face_filename}")
                face_file.save(face_path)
        elif 'face_url' in request.form:
            face_url = request.form['face_url']
            if face_url.startswith(('http://', 'https://')):
                ext = os.path.splitext(urlparse(face_url).path)[1]
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.mp4']:
                    face_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
                    response = requests.get(face_url)
                    if response.status_code == 200:
                        with open(face_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        return jsonify({'error': 'Failed to download face image/video from URL'}), 400
                else:
                    return jsonify({'error': 'Invalid face file type from URL'}), 400
        else:
            return jsonify({'error': 'No face file or URL provided'}), 400

        # Handle audio input
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and allowed_file(audio_file.filename):
                audio_filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio_filename}")
                audio_file.save(audio_path)
        elif 'audio_url' in request.form:
            audio_url = request.form['audio_url']
            if audio_url.startswith(('http://', 'https://')):
                ext = os.path.splitext(urlparse(audio_url).path)[1]
                if ext.lower() in ['.wav', '.mp3']:
                    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
                    response = requests.get(audio_url)
                    if response.status_code == 200:
                        with open(audio_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        return jsonify({'error': 'Failed to download audio from URL'}), 400
                else:
                    return jsonify({'error': 'Invalid audio file type from URL'}), 400
        else:
            return jsonify({'error': 'No audio file or URL provided'}), 400

        # Process the files
        try:
            result_path = process_files(face_path, audio_path)
            result_filename = os.path.basename(result_path)
            
            # Clean up input files
            if face_path and os.path.exists(face_path):
                os.remove(face_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                
            return jsonify({
                'status': 'success',
                'message': 'Processing completed successfully',
                'data': {
                    'video_path': result_path,
                    'download_url': get_full_url(f'/download/{result_filename}'),
                    'preview_url': get_full_url(f'/download/{result_filename}?view=true')
                }
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Processing failed: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Request failed: {str(e)}'
        }), 400

# Health check endpoint
@app.route('/api/health', methods=['GET'])
@require_api_key
def health_check():
    return jsonify({
        'status': 'success',
        'message': 'Service is healthy',
        'data': {
            'version': '1.0'
        }
    }), 200

@app.route('/api/generate', methods=['POST'])
@require_api_key
def generate():
    """Handle both file uploads and URLs for generation"""
    face_path = None
    audio_path = None
    temp_files = []
    
    try:
        # Handle face input (file or URL)
        if 'face' in request.files:
            face_file = request.files['face']
            if face_file and allowed_file(face_file.filename, 
                Config.ALLOWED_IMAGE_EXTENSIONS | Config.ALLOWED_VIDEO_EXTENSIONS):
                face_path = save_uploaded_file(face_file, 'face')['path']
                temp_files.append(face_path)
        elif 'face_url' in request.form:
            face_url = request.form['face_url']
            face_path, error = download_from_url(
                face_url, 
                Config.TEMP_FOLDER,
                Config.ALLOWED_IMAGE_EXTENSIONS | Config.ALLOWED_VIDEO_EXTENSIONS
            )
            if error:
                return jsonify({'error': f'Face URL error: {error}'}), 400
            temp_files.append(face_path)
        else:
            return jsonify({'error': 'No face file or URL provided'}), 400

        # Handle audio input (file or URL)
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and allowed_file(audio_file.filename, Config.ALLOWED_AUDIO_EXTENSIONS):
                audio_path = save_uploaded_file(audio_file, 'audio')['path']
                temp_files.append(audio_path)
        elif 'audio_url' in request.form:
            audio_url = request.form['audio_url']
            audio_path, error = download_from_url(
                audio_url,
                Config.TEMP_FOLDER,
                Config.ALLOWED_AUDIO_EXTENSIONS
            )
            if error:
                return jsonify({'error': f'Audio URL error: {error}'}), 400
            temp_files.append(audio_path)
        else:
            return jsonify({'error': 'No audio file or URL provided'}), 400

        # Convert audio to WAV if needed
        if not audio_path.lower().endswith('.wav'):
            wav_path = convert_audio_to_wav(audio_path)
            temp_files.append(wav_path)
            audio_path = wav_path

        # Generate output path
        output_filename = f"result_{uuid.uuid4()}.mp4"
        output_path = os.path.join(Config.RESULTS_FOLDER, output_filename)

        try:
            # Run inference
            run_inference(face_path, audio_path, output_path)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': 'Media not compatible (Kindly select or upload another photo/video)',
                'details': str(e)
            }), 400

        # Return success response with full URL
        return jsonify({
            'status': 'success',
            'message': 'Video generated successfully',
            'data': {
                'video_path': output_path,
                'download_url': get_full_url(f'/download/{output_filename}'),
                'preview_url': get_full_url(f'/download/{output_filename}?view=true')
            }
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Cleanup error: {e}")

# Add download endpoint for retrieving saved videos
@app.route('/download/<path:filename>')
def download_file(filename):
    """
    Handle file downloads and video streaming with proper range requests
    """
    try:
        file_path = os.path.join(Config.RESULTS_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        file_size = os.path.getsize(file_path)
        
        # Handle video preview/streaming
        if request.args.get('view') == 'true':
            # Get the range header if present
            range_header = request.headers.get('Range', None)
            
            # Default to sending entire file
            byte1, byte2 = 0, file_size - 1
            
            # Parse range header if present
            if range_header:
                match = re.search(r'bytes=(\d+)-(\d*)', range_header)
                if match:
                    groups = match.groups()
                    
                    if groups[0]: 
                        byte1 = int(groups[0])
                    if groups[1] and groups[1].isdigit(): 
                        byte2 = min(int(groups[1]), file_size - 1)

            chunk_size = byte2 - byte1 + 1
            
            # Open the file and seek to the correct position
            with open(file_path, 'rb') as f:
                f.seek(byte1)
                chunk = f.read(chunk_size)
            
            # Create response with the file chunk
            response = Response(
                chunk,
                206 if range_header else 200,
                mimetype='video/mp4',
                direct_passthrough=True
            )
            
            # Set content range header
            response.headers.update({
                'Content-Range': f'bytes {byte1}-{byte2}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(chunk_size),
                'Content-Type': 'video/mp4'
            })

            # Add CORS and caching headers
            response.headers.update({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Range',
                'Cache-Control': 'public, max-age=31536000',
                'Content-Disposition': 'inline'
            })
            
            return response

        # Handle regular downloads
        return send_file(
            file_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        app.logger.error(f"Error in download_file: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.after_request
def after_request(response):
    response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Type, X-API-Key',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges',
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "default-src 'self' blob:; media-src 'self' blob: *;"
    })
    return response

@app.route('/api/keys/generate', methods=['POST'])
@require_admin_key
def generate_api_key():
    """Generate a new API key - admin only"""
    new_key = secrets.token_hex(32)
    return jsonify({
        'api_key': new_key,
        'message': 'Store this key securely - it won\'t be shown again'
    })

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Range')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response

if __name__ == '__main__':
    # Print the API key when starting the server
    print(f"API Key: {Config.API_KEY}")
    
    app.run(host='0.0.0.0', port=5000)