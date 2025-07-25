
import os
from flask import Flask, request, render_template, redirect, url_for, flash
import traceback
# We will import all the functions from our professional search script
from professional_search import (
    get_clip_model, 
    extract_clip_features, 
    download_video, 
    find_and_filter_videos,
    YOUTUBE_API_KEY,
    VIDEOS_TO_CHECK,
    MAXIMUM_VIEW_COUNT
)

# --- Flask App Setup ---
app = Flask(__name__)
DOWNLOADS_FOLDER = 'downloads'
app.config['DOWNLOADS_FOLDER'] = DOWNLOADS_FOLDER
app.secret_key = 'a_very_secret_key_for_your_app'

# --- Load the AI Model ONCE when the server starts ---
# This is slow, so we do it here to avoid reloading for every user.
try:
    print("--- Initializing application and loading AI model... ---")
    if not os.path.exists(DOWNLOADS_FOLDER):
        os.makedirs(DOWNLOADS_FOLDER)
    
    clip_model, clip_preprocess, device = get_clip_model()
    
    print("✅ Application Ready. AI Model is loaded.")
except Exception as e:
    print(f"\n!!! FATAL ERROR: COULD NOT LOAD AI MODEL: {e}!!!")
    clip_model = None # Set to None if loading fails

# --- Web Routes ---
@app.route('/', methods=['GET'])
def index():
    """Handles loading the homepage."""
    if clip_model is None:
        return "<h1>Error</h1><p>The AI model could not be loaded. Please check the server console.</p>", 500
    return render_template('live_search_index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the URL submission and the entire search process."""
    if clip_model is None:
        return "<h1>Error</h1><p>The AI model is not available. Search cannot be performed.</p>", 500

    video_url = request.form.get('video_url')
    if not video_url:
        flash('Please enter a YouTube URL.')
        return redirect(url_for('index'))

    # --- Parse URL ---
    target_video_id = None
    try:
        if "/shorts/" in video_url: target_video_id = video_url.split("/shorts/")[1].split("?")[0]
        elif "v=" in video_url: target_video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url: target_video_id = video_url.split("youtu.be/")[1].split("?")[0]
    except IndexError: pass
    
    if not target_video_id:
        flash('Invalid or unsupported YouTube URL format.')
        return redirect(url_for('index'))

    # --- Run the entire search logic from professional_search.py ---
    print(f"--- Starting live search for Video ID: {target_video_id} ---")
    
    # 1. Analyze Original Video
    original_video_path = download_video(target_video_id, "Original Video")
    if not original_video_path:
        flash('Could not download the original video from the URL provided.')
        return redirect(url_for('index'))
    original_features = extract_clip_features(original_video_path, clip_model, clip_preprocess, device)
    os.remove(original_video_path) # Clean up immediately
    if original_features is None:
        flash('Could not analyze the original video.')
        return redirect(url_for('index'))
        
    # 2. Find and Filter Candidates
    similar_videos = find_and_filter_videos(YOUTUBE_API_KEY, target_video_id, VIDEOS_TO_CHECK, MAXIMUM_VIEW_COUNT)
    
    # 3. Analyze Candidates and Compare
    candidate_results = []
    if similar_videos:
        for video in similar_videos:
            if video['id'] == target_video_id: continue
            candidate_path = download_video(video['id'], video['title'])
            if candidate_path and os.path.exists(candidate_path):
                candidate_features = extract_clip_features(candidate_path, clip_model, clip_preprocess, device)
                if candidate_features is not None:
                    sim = cosine_similarity(original_features.reshape(1,-1), candidate_features.reshape(1,-1))[0][0]
                    candidate_results.append({'id': video['id'], 'title': video['title'], 'similarity': sim})
                os.remove(candidate_path)
    
    # 4. Prepare and Render Results
    if candidate_results:
        sorted_results = sorted(candidate_results, key=lambda x: x['similarity'], reverse=True)
        # Add extra info for the template
        for result in sorted_results:
            result['similarity_percent'] = f"{(result['similarity'] * 100):.2f}%"
            result['url'] = f"https://www.youtube.com/watch?v={result['id']}"
            result['thumbnail'] = f"https://img.youtube.com/vi/{result['id']}/mqdefault.jpg"
        
        return render_template('live_search_results.html', results=sorted_results)
    else:
        flash("Could not find any visually similar videos that met the filter criteria.")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)