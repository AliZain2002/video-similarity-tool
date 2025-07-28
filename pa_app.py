# pa_app.py (PythonAnywhere Version)

import os
from flask import Flask, request, render_template, redirect, url_for, flash
import traceback
# We will use the logic from professional_search.py, but need to redefine the download function
from professional_search import (
    get_clip_model, 
    extract_clip_features, 
    find_and_filter_videos,
    YOUTUBE_API_KEY,
    VIDEOS_TO_CHECK,
    MAXIMUM_VIEW_COUNT
)
import yt_dlp # We need to import this here to use it

# --- Flask App Setup ---
app = Flask(__name__)
# On PythonAnywhere, we use a temporary directory provided
DOWNLOADS_FOLDER = '/tmp' 
app.config['DOWNLOADS_FOLDER'] = DOWNLOADS_FOLDER
app.secret_key = 'a_very_secret_key_for_your_app_on_pa'

# --- NEW PythonAnywhere-safe download function ---
def download_video_pa(video_id, title="video"):
    """Downloads a pre-merged MP4 that does not require FFmpeg."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"\nDownloading (PA-Safe): {title} (ID: {video_id})")
    
    ydl_opts = {
        # THIS IS THE KEY CHANGE: Ask for the best pre-merged MP4 file.
        'format': 'best[ext=mp4]/best', 
        'outtmpl': os.path.join(app.config['DOWNLOADS_FOLDER'], f'{video_id}.%(ext)s'),
        'quiet': True, 'no_warnings': True, 'age_limit': 99
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            return ydl.prepare_filename(info)
    except Exception:
        return None

# --- Load the AI Model ONCE when the server starts ---
try:
    print("--- Initializing application and loading AI model... ---")
    clip_model, clip_preprocess, device = get_clip_model()
    print("✅ Application Ready. AI Model is loaded.")
except Exception as e:
    print(f"\n!!! FATAL ERROR: COULD NOT LOAD AI MODEL: {e}!!!")
    clip_model = None

# --- Web Routes ---
@app.route('/', methods=['GET'])
def index():
    if clip_model is None:
        return "<h1>Error</h1><p>The AI model could not be loaded. Please check the server logs.</p>", 500
    return render_template('live_search_index.html')

@app.route('/search', methods=['POST'])
def search():
    # ... (This function is very similar, but calls our new download function)
    if clip_model is None:
        return "<h1>Error</h1><p>The AI model is not available.</p>", 500

    video_url = request.form.get('video_url')
    if not video_url:
        flash('Please enter a YouTube URL.')
        return redirect(url_for('index'))

    target_video_id = None
    try:
        if "/shorts/" in video_url: target_video_id = video_url.split("/shorts/")[1].split("?")[0]
        elif "v=" in video_url: target_video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url: target_video_id = video_url.split("youtu.be/")[1].split("?")[0]
    except IndexError: pass
    
    if not target_video_id:
        flash('Invalid or unsupported YouTube URL format.')
        return redirect(url_for('index'))

    print(f"--- Starting live search for Video ID: {target_video_id} ---")
    
    # Use our new PythonAnywhere-safe download function
    original_video_path = download_video_pa(target_video_id, "Original Video")
    if not original_video_path:
        flash('Could not download the original video from the URL provided.')
        return redirect(url_for('index'))
        
    original_features = extract_clip_features(original_video_path, clip_model, clip_preprocess, device)
    os.remove(original_video_path)
    if original_features is None:
        flash('Could not analyze the original video.')
        return redirect(url_for('index'))
        
    similar_videos = find_and_filter_videos(YOUTUBE_API_KEY, target_video_id, VIDEOS_TO_CHECK, MAXIMUM_VIEW_COUNT)
    
    candidate_results = []
    if similar_videos:
        for video in similar_videos:
            if video['id'] == target_video_id: continue
            
            # Use our new PythonAnywhere-safe download function
            candidate_path = download_video_pa(video['id'], video['title'])
            if candidate_path and os.path.exists(candidate_path):
                candidate_features = extract_clip_features(candidate_path, clip_model, clip_preprocess, device)
                if candidate_features is not None:
                    sim = cosine_similarity(original_features.reshape(1,-1), candidate_features.reshape(1,-1))[0][0]
                    candidate_results.append({'id': video['id'], 'title': video['title'], 'similarity': sim})
                os.remove(candidate_path)
    
    if candidate_results:
        sorted_results = sorted(candidate_results, key=lambda x: x['similarity'], reverse=True)
        for result in sorted_results:
            result['similarity_percent'] = f"{(result['similarity'] * 100):.2f}%"
            result['url'] = f"https://www.youtube.com/watch?v={result['id']}"
            result['thumbnail'] = f"https://img.youtube.com/vi/{result['id']}/mqdefault.jpg"
        return render_template('live_search_results.html', results=sorted_results)
    else:
        flash("Could not find any visually similar videos that met the filter criteria.")
        return redirect(url_for('index'))