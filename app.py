# app.py (Hugging Face Gradio Version)

import gradio as gr
import os
import yt_dlp
from googleapiclient.discovery import build
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel # Using the stable Hugging Face model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# --- CONFIGURATION ---
# IMPORTANT: On Hugging Face, we use "Secrets" for the API key, not a hardcoded string.
# Go to your Space's "Settings" tab and add a Secret named YOUTUBE_API_KEY
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "DEFAULT_KEY_IF_NOT_SET")

VIDEOS_TO_CHECK = 15
MAXIMUM_VIEW_COUNT = 50000

# --- AI Model Setup ---
# This will happen ONCE when the Space starts.
print("--- Loading Hugging Face Vision Transformer model... ---")
MODEL_NAME = "google/vit-base-patch16-224-in21k"
PROCESSOR = ViTImageProcessor.from_pretrained(MODEL_NAME)
MODEL = ViTModel.from_pretrained(MODEL_NAME)
print("--- Hugging Face model loaded successfully. ---")

# --- Helper Functions (unchanged logic) ---
def download_video(video_id, title="video"):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    out_path = os.path.join("downloads", f'{video_id}.mp4')
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': out_path, 'quiet': True, 'no_warnings': True, 'age_limit': 99}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return out_path
    except Exception:
        return None

def extract_hf_features(video_path, model, processor):
    import cv2
    try:
        video = cv2.VideoCapture(video_path)
        features = []
        while True:
            success, frame = video.read()
            if not success: break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            inputs = processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            features.append(outputs.pooler_output.cpu().numpy().flatten())
        video.release()
        return np.mean(features, axis=0) if features else None
    except Exception:
        return None

def find_and_filter_videos(api_key, video_id, max_results, max_views):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_response = youtube.videos().list(part='snippet,topicDetails', id=video_id).execute()
        candidate_ids = []
        if video_response.get('items'):
            topic_ids = video_response['items'][0].get('topicDetails', {}).get('topicIds', [])
            if topic_ids:
                search_response = youtube.search().list(part='id', topicId=topic_ids[0], type='video', maxResults=max_results).execute()
                candidate_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            if not candidate_ids:
                original_title = video_response['items'][0]['snippet']['title']
                search_response = youtube.search().list(part='id', q=original_title, type='video', maxResults=max_results).execute()
                candidate_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        if not candidate_ids: return []
        video_details_response = youtube.videos().list(part="snippet,statistics", id=",".join(candidate_ids)).execute()
        filtered_videos = []
        for item in video_details_response.get("items", []):
            view_count = int(item.get("statistics", {}).get("viewCount", 0))
            if view_count <= max_views:
                filtered_videos.append({'id': item['id'], 'title': item['snippet']['title'], 'thumbnail': f"https://img.youtube.com/vi/{item['id']}/mqdefault.jpg"})
        return filtered_videos
    except Exception:
        traceback.print_exc()
        return []

# --- The Main Function that Gradio will run ---
def find_hidden_gems(url, progress=gr.Progress(track_tqdm=True)):
    if not url:
        return "Please enter a YouTube URL.", None
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
        
    target_video_id = None
    try:
        if "/shorts/" in url: target_video_id = url.split("/shorts/")[1].split("?")[0]
        elif "v=" in url: target_video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url: target_video_id = url.split("youtu.be/")[1].split("?")[0]
    except IndexError: pass
    if not target_video_id:
        return "Invalid YouTube URL format.", None

    progress(0, desc="Analyzing Original Video...")
    original_path = download_video(target_video_id)
    if not original_path:
        return "Could not download the original video.", None
    original_features = extract_hf_features(original_path, MODEL, PROCESSOR)
    os.remove(original_path)
    if original_features is None:
        return "Could not analyze the original video.", None
    
    progress(0.2, desc="Finding Candidate Videos...")
    similar_videos = find_and_filter_videos(YOUTUBE_API_KEY, target_video_id, VIDEOS_TO_CHECK, MAXIMUM_VIEW_COUNT)
    
    candidate_results = []
    if similar_videos:
        for video in tqdm(similar_videos, desc="Analyzing Hidden Gems"):
            if video['id'] == target_video_id: continue
            candidate_path = download_video(video['id'], video['title'])
            if candidate_path:
                candidate_features = extract_hf_features(candidate_path, MODEL, PROCESSOR)
                if candidate_features is not None:
                    sim = cosine_similarity(original_features.reshape(1,-1), candidate_features.reshape(1,-1))[0][0]
                    candidate_results.append({'title': video['title'], 'similarity': sim, 'thumbnail': video['thumbnail'], 'url': f"https://www.youtube.com/watch?v={video['id']}"})
                os.remove(candidate_path)

    if not candidate_results:
        return "Could not find any visually similar videos that met the filter criteria.", None

    sorted_results = sorted(candidate_results, key=lambda x: x['similarity'], reverse=True)
    
    # Format the output for Gradio's Gallery component
    gallery_output = [(res['thumbnail'], f"{res['title']}\nSimilarity: {(res['similarity'] * 100):.2f}%") for res in sorted_results]
    
    return f"Found {len(sorted_results)} hidden gems!", gallery_output

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🎬 Visual "Hidden Gems" Finder')
    gr.Markdown("Paste a YouTube URL to find visually similar, undiscovered videos.")
    
    with gr.Row():
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        submit_button = gr.Button("Search")

    status_output = gr.Textbox(label="Status", interactive=False)
    gallery_output = gr.Gallery(label="Search Results", show_label=True, elem_id="gallery", columns=[5], rows=[2], object_fit="contain", height="auto")

    submit_button.click(
        fn=find_hidden_gems,
        inputs=url_input,
        outputs=[status_output, gallery_output]
    )

demo.launch()