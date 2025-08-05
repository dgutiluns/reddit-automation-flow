from flask import Flask, request, jsonify, send_file
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os
import subprocess
import uuid
import whisper
import json
import os
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.cloud import storage
import pickle
import cv2
from PIL import Image
import tempfile

app = Flask(__name__)
base_dir = os.path.dirname(__file__)

# --- Video Duplication Detection ---

# Configuration
PROCESSED_STORIES_FILE = "processed_stories.json"
MAX_STORIES_TO_CHECK = 3  # Maximum number of stories to check before giving up

# YOUTUBE CONFIGURATIONS:
# YouTube API Configuration
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_CREDENTIALS_FILE = "youtube_credentials.pickle"  # Store your OAuth2 credentials here

# INSTAGRAM CREDENTIALS:
INSTAGRAM_ACCESS_TOKEN = "IGAARDZCEVjlFRBZAE1zc0FzRHZAsek94cFBuNnp0TkR2VWwtckFoME1Ub2syTEoxM0tiR2c5c3B6M28yMEEzY3ZAaUXF3ZAzUwREJlbGRNc1VlanZA4Q21pX01kc3NTQ0c3TUpMdk12cm5HZAW5FUTRDcDhSaHZARWTFaaUg1UnN6LUxycwZDZD"  # Get this from Facebook Developer Console
INSTAGRAM_ACCOUNT_ID = "30657471253901006"
INSTAGRAM_API_BASE = "https://graph.facebook.com/v18.0"

# GCS CREDENTIALS:

GCS_BUCKET_NAME = "reddit-audio-n8n"


# --- LOADING AND CHECKING PROCESSED STORIES ---
def load_processed_stories():
    """Load the processed stories from JSON file"""
    if os.path.exists(PROCESSED_STORIES_FILE):
        try:
            with open(PROCESSED_STORIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"stories": [], "last_updated": None}
    return {"stories": [], "last_updated": None}

def save_processed_stories(data):
    """Save the processed stories to JSON file"""
    data["last_updated"] = datetime.now().isoformat()
    with open(PROCESSED_STORIES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def is_story_processed(story_id, permalink, url, platform=None):
    """Check if a story has been processed (optionally for a specific platform)"""
    processed_data = load_processed_stories()
    
    for story in processed_data["stories"]:
        if (story.get("id") == story_id or 
            story.get("permalink") == permalink or 
            story.get("url") == url):
            
            if platform is None:
                # If no platform specified, check if processed for any platform
                platforms = story.get("platforms", {})
                return any(platforms.values())
            else:
                # Check specific platform
                platforms = story.get("platforms", {})
                return platforms.get(platform, False)
    return False


def add_processed_story(story_data, platforms=None):
    """Add a story to the processed list with platform tracking"""
    processed_data = load_processed_stories()
    
    if platforms is None:
        platforms = []
    
    # Create a clean record of the processed story
    story_record = {
        "id": story_data.get("id"),
        "permalink": story_data.get("permalink"),
        "url": story_data.get("url"),
        "title": story_data.get("title", "")[:100],
        "author": story_data.get("author"),
        "processed_date": datetime.now().isoformat(),
        "num_comments": story_data.get("num_comments", 0),
        "platforms": {
            "youtube": "youtube" in platforms,
            "instagram": "instagram" in platforms
        }
    }
    
    processed_data["stories"].append(story_record)
    save_processed_stories(processed_data)


@app.route("/check_story", methods=["POST"])
def check_story():
    """
    Check if a single story has been processed
    Expected input: Single story object from Reddit API
    """
    try:
        story_data = request.json
        
        if not story_data:
            return jsonify({"error": "No story data provided"}), 400
        
        story_id = story_data.get("id")
        permalink = story_data.get("permalink")
        url = story_data.get("url")
        
        if not story_id:
            return jsonify({"error": "Story ID is required"}), 400
        
        # Check if story is already processed
        already_processed = is_story_processed(story_id, permalink, url)
        
        response = {
            "story_id": story_id,
            "already_processed": already_processed,
            "story_data": story_data if not already_processed else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Error checking story: {str(e)}"}), 500

@app.route("/find_new_story", methods=["POST"])
def find_new_story():
    """
    Check multiple stories and return the first unprocessed one
    Expected input: Array of story objects from Reddit API
    """
    try:
        # Handle Reddit API format where stories are nested in data.children
        request_data = request.json
        if "data" in request_data and "children" in request_data["data"]:
            # Reddit API format
            stories_data = [child["data"] for child in request_data["data"]["children"]]
        else:
            # Direct stories array format
            stories_data = request_data.get("stories", [])
        
        if not stories_data:
            return jsonify({"error": "No stories data provided"}), 400
        
        checked_count = 0
        
        for story in stories_data[:MAX_STORIES_TO_CHECK]:
            checked_count += 1
            
            story_id = story.get("id")
            permalink = story.get("permalink")
            url = story.get("url")
            
            if not story_id:
                continue
            
            # Check if this story is already processed
            if not is_story_processed(story_id, permalink, url):
                return jsonify({
                    "found_new_story": True,
                    "story_data": story,
                    "checked_count": checked_count,
                    "message": f"Found new story after checking {checked_count} stories"
                })
        
        # If we get here, all stories were already processed
        return jsonify({
            "found_new_story": False,
            "story_data": None,
            "checked_count": checked_count,
            "message": f"All {checked_count} stories have already been processed. Please wait for new content.",
            "suggestion": "Wait for new Reddit posts or increase the number of posts fetched"
        }), 404
        
    except Exception as e:
        return jsonify({"error": f"Error finding new story: {str(e)}"}), 500




@app.route("/mark_story_processed", methods=["POST"])
def mark_story_processed():
    """
    Mark a story as processed for a specific platform or all platforms
    """
    try:
        data = request.json
        story_data = data.get("story_data", {})
        platform = data.get("platform", "all")  # "youtube", "instagram", or "all"
        video_parts = data.get("video_parts", [])  # Add video parts to the input
        base_dir = data.get("base_dir", ".")  # Add base directory to the input
        
        if not story_data or not story_data.get("id"):
            return jsonify({"error": "Story data with ID is required"}), 400
        
        story_id = story_data.get("id")
        
        if platform == "all":
            # Mark as completely processed (both platforms done)
            add_processed_story(story_data)
            # Clean up files only when all platforms are done
            if video_parts:
                for video_filename in video_parts:
                    try:
                        video_path = os.path.join(base_dir, video_filename)
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            print(f"Deleted local file: {video_path}")
                    except Exception as e:
                        print(f"Failed to delete local file {video_path}: {e}")
                        # Continue with other files even if one fails
            
            message = f"Story {story_id} marked as fully processed"

        else:
            # Mark as processed for specific platform only
            # You might want to implement platform-specific tracking here
            # For now, we'll just log it
            print(f"Story {story_id} completed for platform: {platform}")
            message = f"Story {story_id} completed for {platform}"
        
        return jsonify({
            "success": True,
            "message": message,
            "story_id": story_id,
            "platform": platform
        })

    except Exception as e:
        return jsonify({"error": f"Failed to mark story as processed: {str(e)}"}), 500



@app.route("/get_processed_stats", methods=["GET"])
def get_processed_stats():
    """
    Get statistics about processed stories
    """
    try:
        processed_data = load_processed_stories()
        
        return jsonify({
            "total_processed": len(processed_data["stories"]),
            "last_updated": processed_data.get("last_updated"),
            "recent_stories": processed_data["stories"][-5:] if processed_data["stories"] else []
        })
        
    except Exception as e:
        return jsonify({"error": f"Error getting stats: {str(e)}"}), 500

@app.route("/clear_processed_stories", methods=["POST"])
def clear_processed_stories():
    """
    Clear all processed stories (use with caution!)
    Requires confirmation parameter
    """
    try:
        confirmation = request.json.get("confirm_clear", False)
        
        if not confirmation:
            return jsonify({"error": "Confirmation required. Send {\"confirm_clear\": true}"}), 400
        
        # Reset the processed stories file
        save_processed_stories({"stories": []})
        
        return jsonify({
            "success": True,
            "message": "All processed stories cleared"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error clearing processed stories: {str(e)}"}), 500

# ===============================================================================

# --- GOOGLE CLOUD STORAGE HELPER FUNCTIONS ---

def upload_to_gcs_temp(local_file_path, story_id, part_number):
    """Upload video to Google Cloud Storage temporarily and return public URL"""
    try:
        # Initialize the client (assumes you have credentials set up)
        client = storage.Client()
        bucket_name = "reddit-audio-n8n"  # Replace with your bucket name
        bucket = client.bucket(bucket_name)
        
        # Create a temporary filename
        filename = f"temp_instagram/{story_id}_part_{part_number:03d}.mp4"
        blob = bucket.blob(filename)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        
        # Make it publicly accessible (temporarily)
        blob.make_public()
        
        return blob.public_url, filename
        
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")
        return None, None

def delete_from_gcs(bucket_name, filename):
    """Delete temporary file from Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.delete()
        print(f"Deleted {filename} from GCS")
    except Exception as e:
        print(f"Failed to delete {filename} from GCS: {e}")

def upload_to_gcs_temp(local_file_path, story_id, part_number):
    """Upload video to Google Cloud Storage temporarily and return public URL"""
    try:
        # Initialize the client (assumes you have credentials set up)
        client = storage.Client()
        bucket_name = "reddit"  # Replace with your bucket name
        bucket = client.bucket(bucket_name)
        
        # Create a temporary filename
        filename = f"temp_instagram/{story_id}_part_{part_number:03d}.mp4"
        blob = bucket.blob(filename)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        
        # Make it publicly accessible (temporarily)
        blob.make_public()
        
        return blob.public_url, filename
        
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")
        return None, None

def delete_from_gcs(bucket_name, filename):
    """Delete temporary file from Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.delete()
        print(f"Deleted {filename} from GCS")
    except Exception as e:
        print(f"Failed to delete {filename} from GCS: {e}")


# ===============================================================================

# HELPER FUNCTIONS:
def generate_srt(text, duration, max_words=5):
    words = text.split()
    chunk_count = len(words) // max_words + (1 if len(words) % max_words else 0)
    chunk_duration = duration / chunk_count

    srt_lines = []
    for i in range(chunk_count):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)
        line_text = " ".join(words[i*max_words:(i+1)*max_words])
        srt_lines.append(f"{i+1}\n{start_str} --> {end_str}\n{line_text}\n")
    return "".join(srt_lines)

def format_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = (int(seconds) // 3600)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# --- 1. TEXT-TO-SPEECH ---

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text")
    title = data.get("title")
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    # --- REMOVE the OG POST link line ---
    if text.startswith("OG POST:"):
        text = "\n".join(text.splitlines()[1:])

    # Prepend title to the text if title is provided
    if title:
        full_text = f"{title}. {text}"
    else:
        full_text = text
    
    tts = gTTS(full_text)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    return send_file(temp_file.name, mimetype="audio/mpeg", as_attachment=True, download_name="tts.mp3")

# --- 2. GET AUDIO DURATION ---

@app.route("/duration", methods=["POST"])
def get_audio_duration():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Missing audio file"}), 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    audio = AudioSegment.from_file(temp_path)
    duration_sec = len(audio) / 1000.0
    return jsonify({"duration": duration_sec})


# --- 2.5. EXTEND VIDEO ---
@app.route("/extend_video", methods=["POST"])
def extend_video():
    data = request.json
    input_video = data.get("input_video", "resized.mp4")
    duration = data.get("duration")
    output_name = data.get("output_name", "extended.mp4")

    if not duration:
        return jsonify({"error": "Missing 'duration'"}), 400

    input_path = os.path.join(base_dir, input_video)
    output_path = os.path.join(base_dir, output_name)

    try:
        # Get original video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, check=True
        )
        original_duration = float(probe.stdout.strip())

        loop_count = int(float(duration) // original_duration) + 1

        # Create list of input files for concat
        concat_list_path = os.path.join(base_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for _ in range(loop_count):
                f.write(f"file '{input_path}'\n")

        command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path
        ]
        subprocess.run(command, check=True)
        return send_file(output_path, mimetype="video/mp4", as_attachment=True)
    except Exception as e:
        return jsonify({"error": "Extend video failed", "details": str(e)}), 500


# --- 3. ADD CAPTIONS ---

@app.route("/caption", methods=["POST"])
def caption_video():
    audio_url = request.json.get("audio_url")
    input_video = request.json.get("input_video", "resized.mp4")
    output_name = request.json.get("output_name", "captioned.mp4")

    if not audio_url:
        return jsonify({"error": "Missing 'audio_url'"}), 400

    input_path = os.path.join(base_dir, input_video)
    output_path = os.path.join(base_dir, output_name)
    audio_path = os.path.join(base_dir, "input_audio.mp3")

    try:
        print(f"Downloading audio from {audio_url}...")
        subprocess.run(["curl", "-s", "-o", audio_path, audio_url], check=True)
        print("Download complete.")

        print("Loading Whisper model...")
        try:
            model = whisper.load_model("tiny")  # Use "tiny" for faster and lower-memory use
            print("Model loaded. Starting transcription...")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return jsonify({"error": "Failed to load Whisper model", "details": str(e)}), 500

        result = model.transcribe(audio_path)
        print("Transcription done.")

        # Write SRT file
        srt_path = os.path.join(base_dir, "captions.srt")
        with open(srt_path, "w") as f:
            subtitle_index = 1
            for segment in result["segments"]:
                words = segment["text"].strip().split()
                segment_duration = segment["end"] - segment["start"]
                
                # Split into chunks of 3-4 words (adjust this number)
                chunk_size = 5  # Change this to control words per caption
                word_chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
                
                for i, chunk in enumerate(word_chunks):
                    chunk_duration = segment_duration / len(word_chunks)
                    start_time = segment["start"] + (i * chunk_duration)
                    end_time = segment["start"] + ((i + 1) * chunk_duration)
                    
                    start = format_timestamp(start_time)
                    end = format_timestamp(end_time)
                    text = " ".join(chunk)
                    
                    f.write(f"{subtitle_index}\n{start} --> {end}\n{text}\n\n")
                    subtitle_index += 1
                    
        # Overlay subtitles onto video with custom styling (centered, red color)
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"subtitles={srt_path}:force_style='Alignment=10,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BackColour=&H80000000,FontSize=18,FontName=Arial Bold,Shadow=2'",
            "-c:v", "libx264", "-preset", "ultrafast",  # Faster encoding
            "-threads", "0",  # Use all CPU cores
            "-c:a", "copy",
            output_path
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg subtitle error: {e}")
            return jsonify({"error": "FFmpeg subtitle overlay failed", "details": str(e)}), 500

        return send_file(output_path, mimetype="video/mp4", as_attachment=True)
    
    except Exception as e:
        print(f"Unexpected error in /caption: {e}")
        return jsonify({"error": "Whisper captioning failed", "details": str(e)}), 500

# --- 4. COMBINE AUDIO + VIDEO ---
@app.route("/combine", methods=["POST"])
def combine_audio_video():
    data = request.json
    audio_url = data.get("audio_url")
    input_video = data.get("input_video", "captioned.mp4")
    output_name = data.get("output_name", "final_output.mp4")

    if not audio_url:
        return jsonify({"error": "Missing 'audio_url'"}), 400

    audio_file = os.path.join(base_dir, "input_audio.mp3")
    video_file = os.path.join(base_dir, input_video)
    output_file = os.path.join(base_dir, output_name)

    try:
        subprocess.run(["curl", "-s", "-o", audio_file, audio_url], check=True)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_file
        ], check=True)

        return send_file(output_file, mimetype="video/mp4", as_attachment=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Combine failed", "details": str(e)}), 500

# --- 5. SPLIT INTO <60s CLIPS ---
@app.route("/split", methods=["POST"])
def split_video():
    input_video = request.json.get("input_video", "final_output.mp4")
    prefix = str(uuid.uuid4())[:8]
    input_path = os.path.join(base_dir, input_video)

    output_pattern = os.path.join(base_dir, f"{prefix}_part_%03d.mp4")
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", "59",
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]
    try:
        subprocess.run(command, check=True)

        parts = sorted([f for f in os.listdir(base_dir) if f.startswith(f"{prefix}_part_")])
        return jsonify({"parts": parts})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Split failed", "details": str(e)}), 500

# --- YOUTUBE HELPER FUNCTIONS ---

def get_youtube_service():
    """Get authenticated YouTube service"""
    credentials = None
    
    # Load existing credentials
    if os.path.exists(YOUTUBE_CREDENTIALS_FILE):
        with open(YOUTUBE_CREDENTIALS_FILE, 'rb') as token:
            credentials = pickle.load(token)
    
    # If credentials are not valid, refresh them
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise Exception("YouTube credentials not found or invalid. Please run OAuth2 setup.")
    
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)

def generate_youtube_metadata(story_title, part_number, total_parts):
    """Generate YouTube-optimized title, description, and tags"""
    
    # Clean and truncate story title for YouTube (max 100 chars for title)
    clean_title = story_title.replace("AITA", "").replace("aita", "").strip()
    if clean_title.startswith("for "):
        clean_title = clean_title[4:]  # Remove "for " from beginning
    
    # Create title with part info
    title = f"AITA Story - Part {part_number}/{total_parts}: {clean_title}"
    if len(title) > 100:
        # Truncate story title to fit
        max_story_length = 100 - len(f"AITA Story - Part {part_number}/{total_parts}: ")
        clean_title = clean_title[:max_story_length-3] + "..."
        title = f"AITA Story - Part {part_number}/{total_parts}: {clean_title}"
    
    # Create description
    description = f"""Am I The A**hole? - Part {part_number} of {total_parts}

{clean_title}

This is part {part_number} of a {total_parts}-part series. Make sure to watch all parts for the complete story!

What do you think? Is OP the a**hole? Let me know in the comments!

#AITA #AmITheAsshole #RedditStories #Shorts #Drama #Relationships #Family #TrueStory #Storytime #Part{part_number}"""
    
    # Generate tags for better discoverability
    tags = [
        "AITA", "Am I The Asshole", "Reddit Stories", "Reddit", "True Stories",
        "Drama", "Relationships", "Family Drama", "Storytime", "Story",
        "Advice", "Moral Dilemma", "YouTube Shorts", "Shorts", "Viral",
        f"Part {part_number}", "Series", "Multi Part Story"
    ]
    
    # Add contextual tags based on title content
    title_lower = clean_title.lower()
    if any(word in title_lower for word in ["wedding", "marriage", "married"]):
        tags.extend(["Wedding", "Marriage", "Wedding Drama"])
    if any(word in title_lower for word in ["family", "mom", "dad", "parent", "child"]):
        tags.extend(["Family", "Parents", "Family Issues"])
    if any(word in title_lower for word in ["work", "job", "boss", "coworker"]):
        tags.extend(["Work", "Job", "Workplace", "Career"])
    if any(word in title_lower for word in ["friend", "friendship"]):
        tags.extend(["Friendship", "Friends", "Social"])
    
    return title, description, tags


def generate_thumbnail(video_path):
    """Generate thumbnail from first frame of video"""
    try:
        # Use OpenCV to extract first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL image
        img = Image.fromarray(frame_rgb)
        
        # YouTube thumbnail optimal size is 1280x720
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file.name, 'JPEG', quality=95)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"Failed to generate thumbnail: {e}")
        return None


# --- YOUTUBE UPLOADING ROUTE ---
@app.route("/upload_youtube_batch", methods=["POST"])
def upload_youtube_batch():
    """
    Upload all video parts to YouTube
    Expected input: {
        "video_parts": ["path/to/part1.mp4", "path/to/part2.mp4", ...],
        "story_title": "Original story title",
        "base_dir": "/path/to/video/files"
    }
    """
    try:
        data = request.json
        
        video_parts_raw = data.get("video_parts", [])

        # Debug logging
        print(f"DEBUG: Received video_parts type: {type(video_parts_raw)}")
        print(f"DEBUG: Received video_parts content: {video_parts_raw}")

        # Handle different input formats
        if isinstance(video_parts_raw, str):
            try:
                import json
                video_parts = json.loads(video_parts_raw)
            except json.JSONDecodeError:
                video_parts = [part.strip() for part in video_parts_raw.split(',') if part.strip()]
        elif isinstance(video_parts_raw, list):
            video_parts = video_parts_raw
        else:
            return jsonify({"error": f"Invalid video_parts format. Expected list or string, got {type(video_parts_raw)}"}), 400

        # Filter for valid video filenames only
        import re
        valid_video_parts = []
        for part in video_parts:
            if isinstance(part, str) and len(part) > 3:
                if re.match(r'.*\.(mp4|mov|avi|mkv)$', part, re.IGNORECASE):
                    valid_video_parts.append(part)

        video_parts = valid_video_parts
        print(f"DEBUG: Final valid video parts: {video_parts}")

        
        story_title = data.get("story_title", "AITA Story")
        base_dir = data.get("base_dir", ".")
        
        if not video_parts:
            return jsonify({"error": "No valid video files found"}), 400
        
        # Get YouTube service
        try:
            youtube = get_youtube_service()
        except Exception as e:
            return jsonify({"error": f"YouTube authentication failed: {str(e)}"}), 500
        
        total_parts = len(video_parts)
        upload_results = []
        
        for i, video_filename in enumerate(video_parts, 1):
            try:
                # Construct full file path
                video_path = os.path.join(base_dir, video_filename)
                
                if not os.path.exists(video_path):
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": False,
                        "error": f"File not found: {video_path}"
                    })
                    continue
                
                # Generate metadata for this part
                title, description, tags = generate_youtube_metadata(story_title, i, total_parts)
                
                # Prepare upload
                body = {
                    "snippet": {
                        "title": title,
                        "description": description,
                        "tags": tags,
                        "categoryId": "24"  # Entertainment category
                    },
                    "status": {
                        "privacyStatus": "public",  # Change to "private" or "unlisted" if needed
                        "selfDeclaredMadeForKids": False
                    }
                }
                
                # Upload video
                media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
                
                insert_request = youtube.videos().insert(
                    part=",".join(body.keys()),
                    body=body,
                    media_body=media
                )

                
                # Execute upload
                response = insert_request.execute()

                video_id = response.get("id")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Upload thumbnail
                thumbnail_path = generate_thumbnail(video_path)
                if thumbnail_path:
                    try:
                        thumbnail_request = youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
                        )
                        thumbnail_request.execute()
                        print(f"Thumbnail uploaded for part {i}")
                        
                        # Clean up temporary thumbnail file
                        os.unlink(thumbnail_path)
                    except Exception as e:
                        print(f"Failed to upload thumbnail for part {i}: {e}")

                upload_results.append({
                    "part": i,
                    "filename": video_filename,
                    "success": True,
                    "video_id": video_id,
                    "video_url": video_url,
                    "title": title
                })

                print(f"Successfully uploaded part {i}/{total_parts}: {video_url}")


            except Exception as e:
                upload_results.append({
                    "part": i,
                    "filename": video_filename,
                    "success": False,
                    "error": str(e)
                })
                print(f"Failed to upload part {i}: {str(e)}")
        
        # Calculate success rate
        successful_uploads = len([r for r in upload_results if r["success"]])
        
        return jsonify({
            "total_parts": total_parts,
            "successful_uploads": successful_uploads,
            "failed_uploads": total_parts - successful_uploads,
            "results": upload_results,
            "message": f"Uploaded {successful_uploads}/{total_parts} parts to YouTube"
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch upload failed: {str(e)}"}), 500

# INSTAGRAM UPLOADING ROUTE


def generate_instagram_caption(story_title, part_number, total_parts):
    """Generate Instagram-optimized caption with hashtags"""
    base_caption = f"AITA Story - Part {part_number}/{total_parts}\n\n"
    
    # Add story context (truncated for Instagram)
    if len(story_title) > 100:
        story_snippet = story_title[:97] + "..."
    else:
        story_snippet = story_title
    
    caption = base_caption + story_snippet + "\n\n"
    
    # Add relevant hashtags
    hashtags = [
        "#AITA", "#AmITheAsshole", "#Reddit", "#RedditStories", 
        "#Storytime", "#Drama", "#Relationships", "#TrueStory",
        "#Part" + str(part_number), "#Series", "#Viral", "#Trending"
    ]
    
    caption += " ".join(hashtags)
    
    # Instagram caption limit is 2200 characters
    if len(caption) > 2200:
        caption = caption[:2197] + "..."
    
    return caption





@app.route("/upload_instagram_batch", methods=["POST"])
def upload_instagram_batch():
    """Upload all video parts to Instagram as Reels"""
    try:
        data = request.json
        video_parts_raw = data.get("video_parts", [])
        story_title = data.get("story_title", "AITA Story")
        base_dir = data.get("base_dir", ".")
        
        # Debug logging
        print(f"DEBUG: Instagram - Received video_parts type: {type(video_parts_raw)}")
        print(f"DEBUG: Instagram - Received video_parts content: {video_parts_raw}")
        
        # Handle different input formats (same logic as YouTube)
        if isinstance(video_parts_raw, str):
            try:
                import json
                video_parts = json.loads(video_parts_raw)
            except json.JSONDecodeError:
                video_parts = [part.strip() for part in video_parts_raw.split(',') if part.strip()]
        elif isinstance(video_parts_raw, list):
            video_parts = video_parts_raw
        else:
            return jsonify({"error": f"Invalid video_parts format"}), 400
        
        # Filter for valid video filenames
        import re
        valid_video_parts = []
        for part in video_parts:
            if isinstance(part, str) and len(part) > 3:
                if re.match(r'.*\.(mp4|mov)$', part, re.IGNORECASE):
                    valid_video_parts.append(part)
        
        video_parts = valid_video_parts
        print(f"DEBUG: Instagram - Final valid video parts: {video_parts}")
        
        if not video_parts:
            return jsonify({"error": "No valid video files found"}), 400
        
        # Check if Instagram credentials are configured
        if not INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCESS_TOKEN == "your_instagram_access_token":
            return jsonify({"error": "Instagram access token not configured"}), 500
            
        if not INSTAGRAM_ACCOUNT_ID or INSTAGRAM_ACCOUNT_ID == "your_instagram_business_account_id":
            return jsonify({"error": "Instagram account ID not configured"}), 500
        
        total_parts = len(video_parts)
        upload_results = []
        story_id = data.get("story_data", {}).get("id", "unknown")
        
        for i, video_filename in enumerate(video_parts, 1):
            try:
                video_path = os.path.join(base_dir, video_filename)
                
                if not os.path.exists(video_path):
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": False,
                        "error": f"File not found: {video_path}"
                    })
                    continue
                
                # Step 1: Upload to Google Cloud Storage
                print(f"Uploading part {i} to Google Cloud Storage...")
                public_url, gcs_filename = upload_to_gcs_temp(video_path, story_id, i)
                
                if not public_url:
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": False,
                        "error": "Failed to upload to Google Cloud Storage"
                    })
                    continue
                
                # Step 2: Create Instagram media
                caption = generate_instagram_caption(story_title, i, total_parts)
                
                # Create media container
                create_media_url = f"{INSTAGRAM_API_BASE}/{INSTAGRAM_ACCOUNT_ID}/media"
                media_payload = {
                    "media_type": "REELS",
                    "video_url": public_url,
                    "caption": caption,
                    "access_token": INSTAGRAM_ACCESS_TOKEN
                }
                
                print(f"Creating Instagram media container for part {i}...")
                media_response = requests.post(create_media_url, data=media_payload)
                print(f"Media creation response: {media_response.status_code} - {media_response.text}")
                
                if media_response.status_code != 200:
                    # Clean up GCS file
                    delete_from_gcs(GCS_BUCKET_NAME, gcs_filename)  # Use your actual bucket name
                    
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": False,
                        "error": f"Failed to create Instagram media: {media_response.text}"
                    })
                    continue
                
                media_id = media_response.json().get("id")
                
                # Step 3: Wait a moment for media processing
                import time
                time.sleep(2)
                
                # Step 4: Publish the media
                publish_url = f"{INSTAGRAM_API_BASE}/{INSTAGRAM_ACCOUNT_ID}/media_publish"
                publish_payload = {
                    "creation_id": media_id,
                    "access_token": INSTAGRAM_ACCESS_TOKEN
                }
                
                print(f"Publishing Instagram reel for part {i}...")
                publish_response = requests.post(publish_url, data=publish_payload)
                print(f"Publish response: {publish_response.status_code} - {publish_response.text}")
                
                if publish_response.status_code == 200:
                    published_media_id = publish_response.json().get("id")
                    instagram_url = f"https://www.instagram.com/p/{published_media_id}/"
                    
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": True,
                        "media_id": published_media_id,
                        "instagram_url": instagram_url,
                        "caption": caption
                    })
                    
                    print(f"Successfully posted Instagram reel part {i}: {instagram_url}")
                    
                    # Delete local file after successful upload
                    try:
                        os.remove(video_path)
                        print(f"Deleted local file: {video_path}")
                    except Exception as e:
                        print(f"Failed to delete local file {video_path}: {e}")
                        
                else:
                    upload_results.append({
                        "part": i,
                        "filename": video_filename,
                        "success": False,
                        "error": f"Failed to publish Instagram media: {publish_response.text}"
                    })
                
                # Clean up GCS file
                delete_from_gcs(GCS_BUCKET_NAME, gcs_filename)
                
            except Exception as e:
                upload_results.append({
                    "part": i,
                    "filename": video_filename,
                    "success": False,
                    "error": str(e)
                })
                print(f"Failed to upload Instagram part {i}: {str(e)}")
        
        successful_uploads = len([r for r in upload_results if r["success"]])
        
        return jsonify({
            "total_parts": total_parts,
            "successful_uploads": successful_uploads,  
            "failed_uploads": total_parts - successful_uploads,
            "results": upload_results,
            "message": f"Uploaded {successful_uploads}/{total_parts} parts to Instagram"
        })
        
    except Exception as e:
        return jsonify({"error": f"Instagram batch upload failed: {str(e)}"}), 500





if __name__ == "__main__":
    app.run(port=5001)
