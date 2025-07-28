# professional_search.py (FINAL VERSION with "Hidden Gems" Filter & FFmpeg-free)

import os
import yt_dlp
from googleapiclient.discovery import build
import torch
import clip
from PIL import Image
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import colorama

# Initialize colorama
colorama.init(autoreset=True)
Fore = colorama.Fore
Style = colorama.Style

# --- CONFIGURATION ---
YOUTUBE_API_KEY = "AIzaSyAmn8IIfq58VU9AVj14-CYPVNrUUd4ECkA" # MAKE SURE THIS IS CORRECT
VIDEOS_TO_CHECK = 20

# --- "Hidden Gems" Filter ---
MAXIMUM_VIEW_COUNT = 50000 # Find videos with less than 50k views

# --- END CONFIGURATION ---

def get_clip_model():
    print(Fore.YELLOW + "--- Loading CLIP model... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(Fore.GREEN + "--- CLIP model loaded successfully. ---")
    return model, preprocess, device

def extract_clip_features(video_path, model, preprocess, device):
    import cv2
    try:
        video = cv2.VideoCapture(video_path)
        frame_features = []
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success: break
            if frame_count % 30 == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image_input)
                frame_features.append(features.cpu().numpy().flatten())
            frame_count += 1
        video.release()
        if not frame_features: return None
        return np.mean(frame_features, axis=0)
    except Exception:
        return None

def download_video(video_id, title="video"):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    # --- THIS IS THE FFmpeg-free CHANGE ---
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': f'downloads/{video_id}.%(ext)s', 'quiet': True, 'no_warnings': True, 'age_limit': 99}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            return ydl.prepare_filename(info)
    except Exception:
        return None

def find_and_filter_videos(api_key, video__id, max_results, max_views):
    print(f"--- Finding up to {max_results} similar videos for {video_id} ---")
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_response = youtube.videos().list(part='snippet,topicDetails', id=video_id).execute()
        candidate_ids = []
        if video_response.get('items'):
            topic_ids = video_response['items'][0].get('topicDetails', {}).get('topicIds', [])
            if topic_ids:
                print(f" -> Plan A: Searching by topic ID {topic_ids[0]}...")
                search_response = youtube.search().list(part='id', topicId=topic_ids[0], type='video', maxResults=max_results).execute()
                candidate_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            if not candidate_ids:
                print(Fore.YELLOW + " -> Plan A failed. Falling back to Plan B: Searching by title...")
                original_title = video_response['items'][0]['snippet']['title']
                search_response = youtube.search().list(part='id', q=original_title, type='video', maxResults=max_results).execute()
                candidate_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        if not candidate_ids: return []
        print(f"--- Found {len(candidate_ids)} candidates. Fetching details... ---")
        video_details_response = youtube.videos().list(part="snippet,statistics", id=",".join(candidate_ids)).execute()
        filtered_videos = []
        for item in video_details_response.get("items", []):
            view_count = int(item.get("statistics", {}).get("viewCount", 0))
            if view_count <= max_views:
                filtered_videos.append({'id': item['id'], 'title': item['snippet']['title']})
            else:
                print(Fore.RED + f" -> Filtering out '{item['snippet']['title']}' (Views: {view_count} - exceeds maximum)")
        print(Fore.GREEN + f"--- Found {len(filtered_videos)} videos that meet the 'hidden gem' criteria. ---")
        return filtered_videos
    except Exception as e:
        print(Fore.RED + f"\n!!! YouTube API Error: {e} !!!")
        return []

if __name__ == "__main__":
    if YOUTUBE_API_KEY == "YOUR_API_KEY_HERE" or not YOUTUBE_API_KEY:
        print(Fore.RED + "!!! ERROR: Please paste your YouTube API key into the file.")
    else:
        clip_model, clip_preprocess, device = get_clip_model()
        target_url = input("Please enter the YouTube video URL to search for: ")
        target_video_id = None
        try:
            if "/shorts/" in target_url: target_video_id = target_url.split("/shorts/")[1].split("?")[0]
            elif "v=" in target_url: target_video_id = target_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in target_url: target_video_id = target_url.split("youtu.be/")[1].split("?")[0]
        except IndexError: pass
        if not target_video_id:
            print(Fore.RED + "Invalid or unsupported YouTube URL format.")
            exit()
        if not os.path.exists('downloads'): os.makedirs('downloads')
        print("\n" + Fore.CYAN + "--- Analyzing Original Video ---")
        original_video_path = download_video(target_video_id, "Original Video")
        if not original_video_path:
            print(Fore.RED + "\nCould not download the original video. Exiting.")
            exit()
        original_features = extract_clip_features(original_video_path, clip_model, clip_preprocess, device)
        if original_features is None:
            print(Fore.RED + "\nCould not analyze the original video. Exiting.")
            os.remove(original_video_path)
            exit()
        print(" -> Original video analysis complete.")
        similar_videos = find_and_filter_videos(YOUTUBE_API_KEY, target_video_id, VIDEOS_TO_CHECK, MAXIMUM_VIEW_COUNT)
        candidate_results = []
        if similar_videos:
            for video in tqdm(similar_videos, desc=Fore.CYAN + "Analyzing Filtered Videos"):
                if video['id'] == target_video_id: continue
                candidate_path = download_video(video['id'], video['title'])
                if candidate_path and os.path.exists(candidate_path):
                    candidate_features = extract_clip_features(candidate_path, clip_model, clip_preprocess, device)
                    if candidate_features is not None:
                        sim = cosine_similarity(original_features.reshape(1,-1), candidate_features.reshape(1,-1))[0][0]
                        candidate_results.append({'id': video['id'], 'title': video['title'], 'similarity': sim})
                    os.remove(candidate_path)
        os.remove(original_video_path)
        if candidate_results:
            sorted_results = sorted(candidate_results, key=lambda x: x['similarity'], reverse=True)
            print("\n\n" + "="*50)
            print(Style.BRIGHT + Fore.GREEN + "🎉🎉🎉 VISUAL SIMILARITY RANKING COMPLETE 🎉🎉🎉")
            print("="*50)
            for i, result in enumerate(sorted_results):
                similarity_percent = result['similarity'] * 100
                if i == 0:
                    print(Style.BRIGHT + Fore.YELLOW + f"\n🏆 BEST MATCH: Rank #{i+1}")
                    print(Style.BRIGHT + Fore.WHITE + f"   Title: {result['title']}")
                    print(Style.BRIGHT + Fore.GREEN + f"   Similarity: {similarity_percent:.2f}%")
                    print(Style.BRIGHT + Fore.WHITE + f"   URL: https://www.youtube.com/watch?v={result['id']}")
                else:
                    print(Fore.WHITE + f"\n   Rank #{i+1}")
                    print(f"   Title: {result['title']}")
                    print(f"   Similarity: {similarity_percent:.2f}%")
                    print(f"   URL: https://www.youtube.com/watch?v={result['id']}")
            print("\n" + "="*50)
        else:
            print(Fore.YELLOW + "\nCould not find or successfully analyze any videos that met the filter criteria.")
        
        input("\nPress Enter to exit...")