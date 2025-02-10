import os
import numpy as np
from google.cloud import storage, bigquery
from io import BytesIO
from datetime import datetime
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from tqdm import tqdm
import argparse
import joblib
from PIL import Image
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
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype(np.float64)
    lbp_hist /= lbp_hist.sum()
    
    hsv = rgb2hsv(img[:,:,:3])
    hue_hist, _ = np.histogram(hsv[:,:,0], bins=18, range=(0,1))
    saturation_hist, _ = np.histogram(hsv[:,:,1], bins=10, range=(0,1))
    value_hist, _ = np.histogram(hsv[:,:,2], bins=10, range=(0,1))
    
    hue_hist = hue_hist.astype(np.float64) / hue_hist.sum()
    saturation_hist = saturation_hist.astype(np.float64) / saturation_hist.sum()
    value_hist = value_hist.astype(np.float64) / value_hist.sum()
    
    features = np.concatenate((hist, lbp_hist, hue_hist, saturation_hist, value_hist))
    return features

def calculate_quality_index(features, weight, reference_features=None):
    """
    Calculate a quality index based on features and weight, on a scale of 1 to 5.
    Higher weights and less differed features result in values closer to 5.
    Lower weights with more differed features result in values closer to 1.
    """
    features_array = np.array(features, dtype=np.float64)
    
    if features_array.ndim == 0:
        raise ValueError(f"Features are not iterable: {features_array}. Ensure that the feature extraction returned an array.")

    # Normalize features
    if len(features_array) > 0:
        min_val, max_val = features_array.min(), features_array.max()
        if max_val - min_val != 0:
            features_array = (features_array - min_val) / (max_val - min_val)
    
    # Calculate feature difference
    if reference_features is None:
        # If no reference, use the mean of the features as a baseline
        feature_diff = np.mean(np.abs(features_array - np.mean(features_array)))
    else:
        feature_diff = np.mean(np.abs(features_array - reference_features))
    
    # Adjust feature difference based on weight
    # Higher weight reduces the impact of feature difference
    adjusted_diff = feature_diff / (1 + np.log1p(weight))
    
    # Calculate quality index
    # Invert the adjusted difference so that less difference results in higher quality
    quality_index = 5 - (4 * adjusted_diff)
    
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
    
    # Retrieve image file_names and weights from BigQuery
    query = f"""
    SELECT file_name, weight
    FROM `{bq_client.project}.{bq_dataset}.{bq_table}`
    LIMIT {limit}
    """
    query_job = bq_client.query(query)
    results = query_job.result()
    
    image_data = {row.file_name: row.weight for row in results}
    
    if not image_data:
        raise ValueError("No image data found in the input BigQuery table.")
    
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