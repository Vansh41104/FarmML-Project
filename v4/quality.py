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

class ImageProcessor:
    def __init__(self, input_bucket, bq_dataset, bq_table, prefix=None, limit=13550):
        self.input_bucket = input_bucket
        self.bq_dataset = bq_dataset
        self.bq_table = bq_table
        self.prefix = prefix
        self.limit = limit
        self.storage_client = storage.Client()
        self.bq_client = bigquery.Client()
        self.table_id = f"agritwin-cv.njr.metadata"
    
    def load_image_from_gcs(self, image_file):
        """Load a TIF image file from GCS using rasterio."""
        bucket = self.storage_client.bucket(self.input_bucket)
        image_file = image_file.replace('\\', '/')
        blob = bucket.blob(image_file)
        
        if not blob.exists():
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

    def extract_image_features(self, img):
        """Extract various features from an image."""
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray.astype(np.uint8), distances=distances, angles=angles, symmetric=True, normed=True)
        
        features = np.concatenate([
            graycoprops(glcm, 'contrast').ravel(),
            graycoprops(glcm, 'dissimilarity').ravel(),
            graycoprops(glcm, 'homogeneity').ravel(),
            graycoprops(glcm, 'energy').ravel(),
            graycoprops(glcm, 'correlation').ravel(),
            np.histogram(rgb2hsv(img[:, :, :3])[:, :, 0], bins=18, range=(0, 1), density=True)[0],
            np.histogram(rgb2hsv(img[:, :, :3])[:, :, 1], bins=10, range=(0, 1), density=True)[0],
            np.histogram(rgb2hsv(img[:, :, :3])[:, :, 2], bins=10, range=(0, 1), density=True)[0]
        ])
        
        return features

    def calculate_quality_index(self, features, weight, reference_features=None):
        """Calculate a quality index based on features and weight."""
        features_array = np.array(features, dtype=np.float64)
        features_mean = np.mean(features_array)
        features_std = np.std(features_array)
        
        features_normalized = (features_array - features_mean) / features_std if features_std != 0 else features_array - features_mean
        feature_diff = np.mean(np.abs(features_normalized - reference_features)) if reference_features is not None else np.mean(np.abs(features_normalized))
        quality_score = (1 - 1 / (1 + np.exp(-feature_diff))) * (1 + np.log1p(weight) / np.log1p(100))
        
        return max(1, min(1 + 4 * quality_score, 5))

    def add_stage_column_if_not_exists(self):
        """Add the stage column to BigQuery table if it does not exist."""
        query = f"""
        ALTER TABLE {self.table_id}
        ADD COLUMN IF NOT EXISTS stage INT64
        """
        self.bq_client.query(query).result()

    def process_images(self):
        """Process TIF images: extract features, calculate quality indices, and update the BigQuery table."""
        self.add_stage_column_if_not_exists()
        
        query = f"""
        SELECT file_name, weight
        FROM {self.table_id}
        WHERE stage IS NULL OR stage != 2
        LIMIT {self.limit}
        """
        results = self.bq_client.query(query).result()
        image_data = {row.file_name: row.weight for row in results}
        
        if not image_data:
            print("No unprocessed images found.")
            return
        
        reference_features = self.calculate_reference_features(list(image_data.keys())[:10])
        processed_count = 0
        
        for file_name, weight in tqdm(image_data.items(), desc="Processing images"):
            if processed_count >= self.limit:
                break
            try:
                full_path = file_name if file_name.startswith(self.prefix) else os.path.join(self.prefix, file_name)
                img = self.load_image_from_gcs(full_path)
                
                image_features = self.extract_image_features(img)
                quality_index = self.calculate_quality_index(image_features, weight, reference_features)
                
                self.update_bigquery_record(full_path, quality_index, stage=2)
                processed_count += 1
            except FileNotFoundError:
                print(f"File not found: {full_path}")
                self.update_bigquery_record(full_path, stage=1)
            except Exception as e:
                print(f"Error processing {full_path}: {str(e)}")
                self.update_bigquery_record(full_path, stage=1)
        
        self.finalize_unprocessed_records()
        print(f"Processed and updated {processed_count} images successfully.")

    def calculate_reference_features(self, initial_files):
        """Calculate reference features as an average of features from initial images."""
        reference_features = None
        for file_name in initial_files:
            try:
                img = self.load_image_from_gcs(file_name)
                features = self.extract_image_features(img)
                reference_features = features if reference_features is None else reference_features + features
            except Exception:
                continue
        return reference_features / len(initial_files) if reference_features is not None else None

    def update_bigquery_record(self, file_name, quality_index=None, stage=None):
        """Update BigQuery table with the quality index and/or stage for an image."""
        update_query = f"""
        UPDATE {self.table_id}
        SET {", ".join([f"quality_index = {quality_index}" if quality_index is not None else "", f"stage = {stage}" if stage is not None else ""]).strip(", ")}
        WHERE file_name = '{file_name}'
        """
        self.bq_client.query(update_query).result()

    def finalize_unprocessed_records(self):
        """Set stage to 1 for any remaining unprocessed images."""
        final_update_query = f"""
        UPDATE {self.table_id}
        SET stage = 1
        WHERE stage IS NULL
        """
        self.bq_client.query(final_update_query).result()

# def main():
#     parser = argparse.ArgumentParser(description='Process TIF images and update BigQuery table.')
#     parser.add_argument('--input_bucket', type=str, required=True, help='The input bucket name in GCS')
#     parser.add_argument('--bq_dataset', type=str, required=True, help='BigQuery dataset name')
#     parser.add_argument('--bq_table', type=str, required=True, help='BigQuery table name for input and output')
#     parser.add_argument('--prefix', type=str, default='Drone/2022-02-19/DCIM/100FPLAN/RGB', help='Prefix for GCS files')
#     parser.add_argument('--limit', type=int, default=13550, help='Limit the number of images to process')
    
#     args = parser.parse_args()
    
#     processor = ImageProcessor(
#         input_bucket=args.input_bucket,
#         bq_dataset=args.bq_dataset,
#         bq_table=args.bq_table,
#         prefix=args.prefix,
#         limit=args.limit
#     )
#     processor.process_images()

# if _name_ == "_main_":
#     main()