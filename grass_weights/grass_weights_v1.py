import os
import json
from google.cloud import storage
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image
import cv2
import numpy as np
from collections import OrderedDict
import pandas as pd

# Initialize Vertex AI
vertexai.init(project="agritwin-cv", location="us-central1")

def upload_to_gcs(local_file_path, bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"File {local_file_path} uploaded to {bucket_name}/{blob_name}.")
    except Exception as e:
        print(f"Error uploading file to GCS: {str(e)}")

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

def analyze_image_vertex(image_bytes):
    try:
        # Create a Vertex AI Image object
        vertex_image = Image(image_bytes=image_bytes)
        
        # Use MultiModalEmbeddingModel to analyze the image
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        
        # Get the embeddings from the model
        embeddings_response = model.get_embeddings(vertex_image)

        print("Embeddings Response Type:", type(embeddings_response))
        print("Embeddings Response Content:", embeddings_response)

        return embeddings_response
    except Exception as e:
        print(f"Error in Vertex AI image analysis: {str(e)}")
        return None

def extract_vertex_features(embedding_result):
    try:
        print("Embedding Result Type:", type(embedding_result))
        print("Embedding Result Content:", embedding_result.__dict__)
        # Extract relevant features from the embedding_result
        # This is a placeholder - you'll need to implement the actual feature extraction
        return {"vertex_feature": "placeholder"}
    except Exception as e:
        print(f"Error extracting Vertex features: {str(e)}")
        return {}

def preprocess_image(image_bytes):
    try:
        # Convert image bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(f"Decoded image shape: {image.shape}")
        
        # Resize image to 224x224
        image_resized = cv2.resize(image, (224, 224))
        print(f"Resized image shape: {image_resized.shape}")
        
        # Normalize the image by scaling pixel values to [0, 1]
        image_normalized = image_resized / 255.0
        print(f"Normalized image shape: {image_normalized.shape}")
        
        return image_normalized
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return np.zeros((224, 224, 3), dtype=np.float32)

def extract_color_features(image):
    try:
        # Convert image to RGB
        hist_rgb = [cv2.calcHist([image], [i], None, [256], [0, 1]) for i in range(3)]
        
        mean, std = cv2.meanStdDev(image)
        
        return {
            'hist_rgb': [hist.tolist() for hist in hist_rgb],
            'red_mean': float(mean[0]),
            'green_mean': float(mean[1]),
            'blue_mean': float(mean[2]),
            'red_std': float(std[0]),
            'green_std': float(std[1]),
            'blue_std': float(std[2])
        }
    except Exception as e:
        print(f"Error extracting color features: {str(e)}")
        return {}

def extract_shape_features(image):
    print(f"Input image shape to extract_shape_features: {image.shape}")
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Grayscale image shape: {gray.shape}")

        # Calculate edges using Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Edge magnitude
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        print(f"Edge magnitude shape: {edge_magnitude.shape}")

        # Calculate features: area and perimeter
        area = np.sum(edge_magnitude > 0.1)
        perimeter = np.sum(edge_magnitude)

        return {
            'edge_area': float(area),
            'edge_perimeter': float(perimeter),
            'edge_complexity': float(perimeter / (area + 1e-5))
        }
    except Exception as e:
        print(f"Error in extract_shape_features: {str(e)}")
        return {
            'edge_area': 0.0,
            'edge_perimeter': 0.0,
            'edge_complexity': 0.0
        }

def extract_texture_features(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobel X and Y edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        features = {}
        for name, filtered in [('sobel_x', sobelx), ('sobel_y', sobely), ('laplacian', laplacian)]:
            mean, std = cv2.meanStdDev(filtered)
            features[f'{name}_mean'] = float(mean)
            features[f'{name}_std'] = float(std)

        return features
    except Exception as e:
        print(f"Error extracting texture features: {str(e)}")
        return {}

def extract_custom_features(image_bytes):
    try:
        image = preprocess_image(image_bytes)
        print(f"Preprocessed image shape: {image.shape}")
        
        features = OrderedDict()
        
        features['color'] = extract_color_features(image)
        print("Color features extracted successfully")
        
        features['shape'] = extract_shape_features(image)
        print("Shape features extracted successfully")
        
        features['texture'] = extract_texture_features(image)
        print("Texture features extracted successfully")
        
        return features
    except Exception as e:
        print(f"Error in extract_custom_features: {str(e)}")
        return OrderedDict({
            'color': {},
            'shape': {},
            'texture': {}
        })

def summarize_features(vertex_features, custom_features):
    try:
        summary = OrderedDict()
        
        summary['vertex'] = vertex_features
        
        summary['custom'] = OrderedDict()
        for feature_type, feature_dict in custom_features.items():
            summary['custom'][feature_type] = OrderedDict()
            summary['custom'][feature_type]['num_features'] = len(feature_dict)
            
            if feature_type == 'color':
                summary['custom'][feature_type]['histogram_bins'] = len(feature_dict.get('hist_rgb', []))
                summary['custom'][feature_type]['color_statistics'] = ['mean', 'std']
            elif feature_type == 'shape':
                summary['custom'][feature_type]['shape_descriptors'] = list(feature_dict.keys())
            elif feature_type == 'texture':
                summary['custom'][feature_type]['texture_descriptors'] = list(feature_dict.keys())
        
        return summary
    except Exception as e:
        print(f"Error summarizing features: {str(e)}")
        return {}

def process_image(local_file_path, bucket_name, output_dir, csv_file):
    try:
        os.makedirs(output_dir, exist_ok=True)

        blob_name = os.path.basename(local_file_path)
        upload_to_gcs(local_file_path, bucket_name, blob_name)

        image_bytes = load_image_from_gcs(bucket_name, blob_name)
        if image_bytes is None:
            raise ValueError("Failed to load image from GCS")

        # Vertex AI analysis using MultiModalEmbeddingModel
        vertex_analysis = analyze_image_vertex(image_bytes)
        vertex_features = extract_vertex_features(vertex_analysis)

        # Custom feature extraction
        custom_features = extract_custom_features(image_bytes)

        # Summarize all features
        summary = summarize_features(vertex_features, custom_features)

        # Prepare data for CSV
        row_data = {
            'filename': blob_name,
            'vertex_feature': summary['vertex']['vertex_feature'],
            'color_num_features': summary['custom']['color']['num_features'],
            'color_histogram_bins': summary['custom']['color']['histogram_bins'],
            'color_statistics': ", ".join(summary['custom']['color']['color_statistics']),
            'shape_num_features': summary['custom']['shape']['num_features'],
            'shape_descriptors': ", ".join(summary['custom']['shape']['shape_descriptors']),
            'texture_num_features': summary['custom']['texture']['num_features'],
            'texture_descriptors': ", ".join(summary['custom']['texture']['texture_descriptors']),
        }

        # Append data to CSV
        df = pd.DataFrame([row_data])
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

        print(f"Analysis results saved to {csv_file}")

        return summary
    except Exception as e:
        print(f"Error processing image {local_file_path}: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    local_image_dir = "D:/Project Caldermeade/NJR/Training/AG/2022-01-24/DCIM/104FPLAN_20/AG_PNG"
    bucket_name = "njr_paddockbucket"
    output_dir = "D:/Project Caldermeade/NJR/Training/AG/2022-01-24/DCIM/104FPLAN_20/AG_PNG_output"
    csv_file = os.path.join(output_dir, 'analysis_results.csv')

    # Create the CSV file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        headers = [
            'filename', 'vertex_feature', 'color_num_features', 
            'color_histogram_bins', 'color_statistics', 
            'shape_num_features', 'shape_descriptors', 
            'texture_num_features', 'texture_descriptors'
        ]
        pd.DataFrame(columns=headers).to_csv(csv_file, index=False)

    for filename in os.listdir(local_image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            local_file_path = os.path.join(local_image_dir, filename)
            print(f"Processing {filename}...")
            summary = process_image(local_file_path, bucket_name, output_dir, csv_file)
            if summary:
                print(f"Completed analysis for {filename}")
            else:
                print(f"Failed to process {filename}")
            print("---")

    print("All images processed!")
