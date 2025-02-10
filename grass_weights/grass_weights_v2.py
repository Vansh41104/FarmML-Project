import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from google.cloud import storage
from io import BytesIO

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./agritwin-cv-5867208ae5d4.json"

# --------------------------------------- IMAGE FEATURE EXTRACTION ---------------------------------------
def extract_image_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_img)
    variance = (cv2.meanStdDev(gray_img))[0][0] ** 2

    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    mean_edges = np.mean(edges)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    mean_laplacian = np.mean(np.abs(laplacian))
    
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_normalized = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
    
    # LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    red_index = np.mean(R) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    green_index = np.mean(G) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    blue_index = np.mean(B) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    
    features = {
        'Mean_Intensity': mean_intensity,
        'Variance': variance,
        'Mean_Edges': mean_edges,
        'Mean_Laplacian': mean_laplacian,
        'Entropy': entropy,
        'Red_Index': red_index,
        'Green_Index': green_index,
        'Blue_Index': blue_index
    }
    
    # Add LBP histogram features
    for i, value in enumerate(lbp_hist):
        features[f'LBP_Hist_{i}'] = value
    
    return features

# --------------------------------------- GCP FUNCTIONS ---------------------------------------
def list_blobs_in_bucket(bucket_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    return [blob.name for blob in blobs]

def load_image_from_gcs(bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        return image_bytes
    except Exception as e:
        print(f"Error loading image from GCS: {str(e)}")
        return None

# --------------------------------------- SAVE FEATURES TO CSV ---------------------------------------
def save_features_to_csv(features_list, output_bucket_name, output_blob_name):
    df_features = pd.DataFrame(features_list)
    csv_buffer = BytesIO()
    df_features.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(output_bucket_name)
    blob = bucket.blob(output_blob_name)
    blob.upload_from_file(csv_buffer, content_type='text/csv')
    print(f"Features saved to gs://{output_bucket_name}/{output_blob_name}")

# --------------------------------------- PROCESS IMAGES IN A BUCKET ---------------------------------------
def process_images_in_bucket(input_bucket_name, output_bucket_name):
    features_list = []
    image_files = list_blobs_in_bucket(input_bucket_name)

    for image_file in tqdm(image_files, desc="Processing images"):
        # Load image from input GCS bucket
        image_bytes = load_image_from_gcs(input_bucket_name, image_file)

        if image_bytes is None:
            print(f"Error: Could not load the image from GCS for {image_file}.")
            continue

        # Convert image bytes to numpy array for OpenCV processing
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Extract local features
        local_features = extract_image_features(img)

        # Add image file name to features
        local_features['Image_File'] = image_file

        # Append the features to the list
        features_list.append(local_features)

    # Save all features to a CSV file in the output bucket
    save_features_to_csv(features_list, output_bucket_name, 'extracted_image_features.csv')

# --------------------------------------- MAIN FUNCTION ---------------------------------------
if __name__ == "__main__":
    # Define your Google Cloud Storage bucket names
    INPUT_BUCKET_NAME = "njr_paddockbucket"
    OUTPUT_BUCKET_NAME = "njr_paddockbucket_output"

    # Process all images in the input bucket and save results to the output bucket
    process_images_in_bucket(INPUT_BUCKET_NAME, OUTPUT_BUCKET_NAME)