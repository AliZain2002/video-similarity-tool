# Import necessary libraries
import cv2  # For video processing
import numpy as np  # For numerical operations
from tensorflow.keras.applications import VGG16  # A pre-trained model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os

# --- 1. Set up the Pre-trained Model ---

def get_feature_extractor():
    """
    Creates and returns a pre-trained VGG16 model, modified to
    extract features from an intermediate layer.
    """
    # Load the VGG16 model, pre-trained on ImageNet data
    # include_top=False removes the final classification layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # We'll use the output of the final pooling layer as our feature vector
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return model

# --- 2. Process a Single Video ---

def extract_features(video_path, model):
    """
    Extracts a feature vector from a video file.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    video = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        # Process every 15th frame to speed things up
        if frame_count % 15 == 0:
            # Resize and preprocess the frame for the VGG16 model
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Extract features from the frame
            features = model.predict(img, verbose=0) # verbose=0 to hide prediction progress
            frame_features.append(features.flatten())

        frame_count += 1


    video.release()

    if not frame_features:
        print("Could not extract any frames from the video.")
        return None

    # --- 3. Create a Single "Fingerprint" for the Video ---
    # We'll average the features of all processed frames to get one representative vector
    video_features = np.mean(frame_features, axis=0)
    return video_features

# --- Main Execution ---

if __name__ == "__main__":
    # ------------------  IMPORTANT!  ------------------
    # --- Change this to the name of your video file ---
    target_video_path = "my_video.mp4"
    # ----------------------------------------------------

    # Get the feature extraction model
    print("Loading the feature extraction model...")
    feature_extractor = get_feature_extractor()
    print("Model loaded.")


    # Extract the feature vector for your target video
    print(f"Analyzing video: {target_video_path}")
    target_features = extract_features(target_video_path, feature_extractor)

    if target_features is not None:
        print("\n-------------------------------------------")
        print("SUCCESS! Created a feature vector (fingerprint) for the video.")
        print("Feature vector shape:", target_features.shape)
        print("-------------------------------------------")
        # In a full application, you would now compare this vector
        # with a database of other video feature vectors.
    else:
        print("Could not process the video. Please check the file path and video format.")