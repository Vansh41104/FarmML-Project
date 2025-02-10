import os
import numpy as np
from google.cloud import storage, bigquery
from io import BytesIO
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2hsv
from tqdm import tqdm
import argparse
import rasterio

def load_image_from_gcs(bucket_name, image_file):
    """Load a TIF image file from GCS using rasterio."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Normalize the file path
    image_file = image_file.replace('\\', '/')
    
    # Try to find the blob
    blob = bucket.blob(image_file)
    if not blob.exists():
        # If not found, try without the prefix
        image_file = image_file.split('/')[-1]
        blob = bucket.blob(image_file)
        if not blob.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
    
    image_bytes = blob.download_as_bytes()
    
    with rasterio.open(BytesIO(image_bytes)) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
        
        if image.shape[2] > 3:
            image = image[:, :, :3]
        
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
    
    return image

def extract_image_features(img):
    """Extract various features from an image."""
    # Convert to grayscale
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Extract texture features using Haralick texture
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray.astype(np.uint8), distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    
    # Color features
    hsv = rgb2hsv(img[:,:,:3])
    hue_hist, _ = np.histogram(hsv[:,:,0], bins=18, range=(0,1), density=True)
    saturation_hist, _ = np.histogram(hsv[:,:,1], bins=10, range=(0,1), density=True)
    value_hist, _ = np.histogram(hsv[:,:,2], bins=10, range=(0,1), density=True)
    
    # Combine all features
    features = np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation,
        hue_hist, saturation_hist, value_hist
    ])
    
    return features

def calculate_quality_index(features, weight, reference_features=None):
    """
    Calculate a quality index based on features and weight, on a scale of 1 to 5.
    Higher quality images will have values closer to 5, lower quality closer to 1.
    """
    features_array = np.array(features, dtype=np.float64)
    
    if features_array.ndim == 0:
        raise ValueError(f"Features are not iterable: {features_array}. Ensure that the feature extraction returned an array.")

    # Normalize features using z-score normalization
    features_mean = np.mean(features_array)
    features_std = np.std(features_array)
    if features_std != 0:
        features_normalized = (features_array - features_mean) / features_std
    else:
        features_normalized = features_array - features_mean

    # Calculate feature difference
    if reference_features is not None:
        reference_normalized = (reference_features - features_mean) / features_std
        feature_diff = np.mean(np.abs(features_normalized - reference_normalized))
    else:
        # If no reference, use the deviation from the mean
        feature_diff = np.mean(np.abs(features_normalized))

    # Apply a sigmoid function to map the difference to a [0, 1] range
    sigmoid = 1 / (1 + np.exp(-feature_diff))

    # Incorporate weight using a logarithmic scale to reduce its impact
    weight_factor = np.log1p(weight) / np.log1p(100)  # Assuming max weight is 100

    # Calculate initial quality score (0 to 1 range)
    quality_score = (1 - sigmoid) * (1 + weight_factor)

    # Scale the quality score to the range [1, 5]
    quality_index = 1 + 4 * quality_score

    # Ensure the quality index is within [1, 5]
    quality_index = max(1, min(quality_index, 5))

    return quality_index

def process_images(input_bucket_name, bq_dataset, bq_table, prefix=None, limit=13550):
    """Process TIF images: extract features, calculate quality indices, and update the input table."""
    bq_client = bigquery.Client()
    
    # First, add the stage column if it doesn't exist
    add_stage_column_query = f"""
    ALTER TABLE `{bq_client.project}.{bq_dataset}.{bq_table}`
    ADD COLUMN IF NOT EXISTS stage INT64
    """
    bq_client.query(add_stage_column_query).result()
    
    # Retrieve image file_names and weights from BigQuery for unprocessed images
    query = f"""
    SELECT file_name, weight
    FROM `{bq_client.project}.{bq_dataset}.{bq_table}`
    WHERE stage IS NULL OR stage != 2
    LIMIT {limit}
    """
    query_job = bq_client.query(query)
    results = query_job.result()
    
    image_data = {row.file_name: row.weight for row in results}
    
    if not image_data:
        print("No unprocessed images found in the input BigQuery table.")
        return
    
    storage_client = storage.Client()
    table_id = f"{bq_client.project}.{bq_dataset}.{bq_table}"

    # Calculate reference features (you may want to adjust this based on your specific needs)
    reference_features = None
    for file_name in list(image_data.keys())[:10]:  # Use first 10 images to calculate reference
        try:
            full_path = os.path.join(prefix, file_name) if prefix else file_name
            full_path = full_path.replace('\\', '/')
            img = load_image_from_gcs(input_bucket_name, full_path)
            features = extract_image_features(img)
            if reference_features is None:
                reference_features = features
            else:
                reference_features += features
        except Exception:
            continue
    
    if reference_features is not None:
        reference_features /= 10  # Average of the first 10 successfully processed images

    processed_count = 0
    for file_name, weight in tqdm(image_data.items(), desc="Processing images"):
        if processed_count >= limit:
            break
        try:
            file_name = file_name.replace('\\', '/')
            
            if prefix and not file_name.startswith(prefix):
                full_path = os.path.join(prefix, file_name)
            else:
                full_path = file_name

            full_path = full_path.replace('\\', '/')
            
            img = load_image_from_gcs(input_bucket_name, full_path)
            
            image_features = extract_image_features(img)
            quality_index = calculate_quality_index(image_features, weight, reference_features)

            # Set the stage to 2 for successfully processed images
            new_stage = 2

            # Update the existing row in BigQuery
            update_query = f"""
            UPDATE `{table_id}`
            SET quality_index = {quality_index}, stage = {new_stage}
            WHERE file_name = '{full_path}'
            """
            update_job = bq_client.query(update_query)
            update_job.result()  # Wait for the query to complete

            processed_count += 1
        
        except FileNotFoundError as e:
            print(f"File not found: {full_path}. Error: {str(e)}")
            # Set stage to 1 for files not found
            update_query = f"""
            UPDATE `{table_id}`
            SET stage = 1
            WHERE file_name = '{full_path}'
            """
            bq_client.query(update_query).result()
        except Exception as e:
            print(f"Error processing {full_path}: {str(e)}")
            # Set stage to 1 for files that encountered errors
            update_query = f"""
            UPDATE `{table_id}`
            SET stage = 1
            WHERE file_name = '{full_path}'
            """
            bq_client.query(update_query).result()

    print(f"Processed and updated {processed_count} images successfully.")

    # After processing, set stage to 1 for any remaining unprocessed images
    final_update_query = f"""
    UPDATE `{table_id}`
    SET stage = 1
    WHERE stage IS NULL
    """
    bq_client.query(final_update_query).result()


def main():
    """Main function to parse arguments and initiate processing."""
    parser = argparse.ArgumentParser(description='Process TIF images, calculate quality index, and update BigQuery table.')
    parser.add_argument('--input_bucket', type=str, required=True, help='The input bucket name in GCS')
    parser.add_argument('--bq_dataset', type=str, required=True, help='BigQuery dataset name')
    parser.add_argument('--bq_table', type=str, required=True, help='BigQuery table name for input and output')
    parser.add_argument('--prefix', type=str, default='Drone/2022-02-19/DCIM/100FPLAN/RGB', help='Prefix for GCS files')
    parser.add_argument('--limit', type=int, default=13550, help='Limit the number of images to process')
    
    args = parser.parse_args()
    
    process_images(
        input_bucket_name=args.input_bucket,
        bq_dataset=args.bq_dataset,
        bq_table=args.bq_table,
        prefix=args.prefix,
        limit=args.limit
    )

if __name__ == "__main__":
    main()