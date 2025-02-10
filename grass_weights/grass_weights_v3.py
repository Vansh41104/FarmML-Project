import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from PIL import Image
from google.cloud import storage
from io import BytesIO
import argparse
import joblib
import re
from datetime import datetime
from google.cloud import bigquery
# Synthetic data generation
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    features = {
        'Mean_Intensity': np.random.uniform(50, 200, num_samples),
        'Variance': np.random.uniform(10, 150, num_samples),
        'Mean_Edges': np.random.uniform(10, 250, num_samples),
        'Mean_Laplacian': np.random.uniform(1, 20, num_samples),
        'LBP_1': np.random.uniform(0, 0.3, num_samples),
        'LBP_2': np.random.uniform(0, 0.3, num_samples),
        'LBP_3': np.random.uniform(0, 0.3, num_samples),
        'LBP_4': np.random.uniform(0, 0.3, num_samples),
        'LBP_5': np.random.uniform(0, 0.3, num_samples),
        'Entropy': np.random.uniform(0, 10, num_samples),
        'Red_Index': np.random.uniform(0, 1, num_samples),
        'Green_Index': np.random.uniform(0, 1, num_samples),
        'Blue_Index': np.random.uniform(0, 1, num_samples),
        'Grass_Weight': np.random.uniform(0, 50, num_samples)  # Target variable
    }
    
    return pd.DataFrame(features)

# Train the predictive model with enhanced grid search
def train_predictive_model(n_samples=10, test_size=0.2):
    # Generate synthetic data (assuming this function exists)
    df_synthetic = generate_synthetic_data(num_samples=n_samples)
    X = df_synthetic.drop('Grass_Weight', axis=1).values
    y = df_synthetic['Grass_Weight'].values


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Feature engineering
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Define the parameter grid
    param_grid = {
       'n_estimators': [100, 200, 300],
       'learning_rate': [0.01, 0.05, 0.1],
       'max_depth': [3, 4, 5, 6],
       'min_child_weight': [1, 3, 5],
       'subsample': [0.8, 0.9, 1.0],
       'colsample_bytree': [0.8, 0.9, 1.0],
       'gamma': [0, 0.1, 0.2],
       'reg_alpha': [0, 0.1, 0.5],
       'reg_lambda': [0.5, 1.0, 1.5],
   }
   


    # Initialize model
    model = XGBRegressor(
        tree_method='hist',  # Faster tree method
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Grid search
    grid_search = GridSearchCV(
        model, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error',
        n_jobs=-1, 
        cv=2, 
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test R-squared Score: {r2}")

    return best_model, scaler, poly


def extract_image_features(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_pixels = gray_img.size

        # Calculate the sum of pixel intensities for proportion calculation
        total_intensity = np.sum(gray_img)

        if total_intensity == 0:
            total_intensity = 1e-7  # Avoid division by zero

        # Calculate proportions instead of mean values
        intensity_proportion = total_intensity / (255 * total_pixels)  # Max intensity is 255
        variance = (cv2.meanStdDev(gray_img)[1][0][0]) ** 2 / (255**2)  # Normalize variance as a proportion

        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        mean_edges_proportion = np.sum(edges) / (255 * total_pixels)  # Edge intensity proportion

        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        laplacian_proportion = np.sum(np.abs(laplacian)) / (255 * total_pixels)

        # Histogram for entropy calculation
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / (hist.sum() + 1e-7)  # Prevent division by zero
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))  # Entropy stays the same

        # LBP features (unchanged)
        lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize histogram

        # RGB channel proportions
        R = img[:, :, 2].astype(float)
        G = img[:, :, 1].astype(float)
        B = img[:, :, 0].astype(float)
        total_color = np.sum(R) + np.sum(G) + np.sum(B) + 1e-7  # Avoid division by zero
        red_proportion = np.sum(R) / total_color
        green_proportion = np.sum(G) / total_color
        blue_proportion = np.sum(B) / total_color

        features = {
            'Intensity_Proportion': intensity_proportion,
            'Variance_Proportion': variance,
            'Edges_Proportion': mean_edges_proportion,
            'Laplacian_Proportion': laplacian_proportion,
            'LBP_1': lbp_hist[0],
            'LBP_2': lbp_hist[1],
            'LBP_3': lbp_hist[2],
            'LBP_4': lbp_hist[3],
            'LBP_5': lbp_hist[4],
            'Entropy': entropy,
            'Red_Proportion': red_proportion,
            'Green_Proportion': green_proportion,
            'Blue_Proportion': blue_proportion
        }

        # Validate and handle non-numeric values
        for key, value in features.items():
            if not isinstance(value, (int, float, np.float32, np.float64)) or np.isnan(value) or np.isinf(value):
                print(f"Non-numeric or invalid value detected in feature {key}: {value}. Replacing with np.nan.")
                features[key] = np.nan  # Replace invalid values with NaN

        return features

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Return default NaN values if there's an error in processing
        return {
            'Intensity_Proportion': np.nan,
            'Variance_Proportion': np.nan,
            'Edges_Proportion': np.nan,
            'Laplacian_Proportion': np.nan,
            'LBP_1': np.nan,
            'LBP_2': np.nan,
            'LBP_3': np.nan,
            'LBP_4': np.nan,
            'LBP_5': np.nan,
            'Entropy': np.nan,
            'Red_Proportion': np.nan,
            'Green_Proportion': np.nan,
            'Blue_Proportion': np.nan
        }



# Add GCS functions
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

def save_features_to_csv(features_list, output_bucket_name, output_blob_name):
    try:
        # Convert the features list to a DataFrame
        df_features = pd.DataFrame(features_list)
        
        # Check if DataFrame is empty
        if df_features.empty:
            print("No features to save. The DataFrame is empty.")
            return
        
        # Write DataFrame to a CSV buffer
        csv_buffer = BytesIO()
        df_features.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer position to the beginning

        # Upload CSV to GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket_name)
        blob = bucket.blob(output_blob_name)
        blob.upload_from_file(csv_buffer, content_type='text/csv')
        
        print(f"Features saved to gs://{output_bucket_name}/{output_blob_name}")
    except Exception as e:
        print(f"Error saving features to CSV: {str(e)}")


# Main image processing function
# Main image processing function
# Main image processing function
def process_images_in_bucket(input_bucket_name, output_bucket_name, model, scaler, poly, slice_number, patch_number, paddock_number, bq_dataset, bq_table):
    features_list = []
    image_files = list_blobs_in_bucket(input_bucket_name)
    
    # Create BigQuery client
    bq_client = bigquery.Client()
    table_id = f"{bq_client.project}.{bq_dataset}.{bq_table}"

    for image_file in tqdm(image_files, desc="Processing images"):
        image_bytes = load_image_from_gcs(input_bucket_name, image_file)
        if image_bytes is None:
            print(f"Error: Could not load the image from GCS for {image_file}.")
            continue

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        local_features = extract_image_features(img)
        local_features['Image_File'] = image_file
        
        # Use current date and time in GCP compatible format
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        local_features['Image_Date'] = current_datetime

        # Add slice, patch, and paddock numbers
        local_features['Slice_Number'] = slice_number
        local_features['Patch_Number'] = patch_number
        local_features['Paddock_Number'] = paddock_number

        # Predict grass weight
        feature_values = [
            local_features['Intensity_Proportion'], 
            local_features['Variance_Proportion'], 
            local_features['Edges_Proportion'],
            local_features['Laplacian_Proportion'], 
            local_features['LBP_1'], local_features['LBP_2'],
            local_features['LBP_3'], local_features['LBP_4'], 
            local_features['LBP_5'],
            local_features['Entropy'], 
            local_features['Red_Proportion'], 
            local_features['Green_Proportion'],
            local_features['Blue_Proportion']
        ]

        feature_values = [0 if (value is None or np.isnan(value) or np.isinf(value)) else value for value in feature_values]

        X = np.array([feature_values])
        X_poly = poly.transform(X)
        X_scaled = scaler.transform(X_poly)

        predicted_grass_weight = model.predict(X_scaled)[0]
        local_features['Predicted_Grass_Weight'] = float(predicted_grass_weight)  # Convert to Python float

        features_list.append(local_features)

        # Prepare row for BigQuery
        bq_row = {
            'image_file': local_features['Image_File'],
            'image_date': local_features['Image_Date'],
            'slice_number': local_features['Slice_Number'],
            'patch_number': local_features['Patch_Number'],
            'paddock_number': local_features['Paddock_Number'],
            'predicted_grass_weight': local_features['Predicted_Grass_Weight']
        }

        # Insert row into BigQuery
        errors = bq_client.insert_rows_json(table_id, [bq_row])
        if errors:
            print(f"Encountered errors while inserting row: {errors}")

    save_features_to_csv(features_list, output_bucket_name, 'extracted_image_features_with_predictions.csv')


def save_model_to_gcs(model, scaler, poly, bucket_name, model_prefix):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Save model
            model_buffer = BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            model_blob = bucket.blob(f"{model_prefix}_model.joblib")
            model_blob.upload_from_file(model_buffer, content_type='application/octet-stream')

            # Save scaler
            scaler_buffer = BytesIO()
            joblib.dump(scaler, scaler_buffer)
            scaler_buffer.seek(0)
            scaler_blob = bucket.blob(f"{model_prefix}_scaler.joblib")
            scaler_blob.upload_from_file(scaler_buffer, content_type='application/octet-stream')

            # Save polynomial features
            poly_buffer = BytesIO()
            joblib.dump(poly, poly_buffer)
            poly_buffer.seek(0)
            poly_blob = bucket.blob(f"{model_prefix}_poly.joblib")
            poly_blob.upload_from_file(poly_buffer, content_type='application/octet-stream')

            print(f"Model, scaler, and polynomial features saved to gs://{bucket_name}/{model_prefix}_*.joblib")
        except Exception as e:
            print(f"Error saving model to GCS: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a GCS bucket, predict grass weight, and upload to BigQuery.')
    parser.add_argument('--input_bucket', type=str, required=True, help='The input bucket name in GCS')
    parser.add_argument('--output_bucket', type=str, required=True, help='The output bucket name in GCS')
    parser.add_argument('--slice_number', type=int, required=True, help='Slice number')
    parser.add_argument('--patch_number', type=int, required=True, help='Patch number')
    parser.add_argument('--paddock_number', type=int, required=True, help='Paddock number')
    parser.add_argument('--bq_dataset', type=str, required=True, help='BigQuery dataset name')
    parser.add_argument('--bq_table', type=str, required=True, help='BigQuery table name')
    parser.add_argument('--model_bucket', type=str, required=True, help='GCS bucket to save the model')
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix for saved model files')

    args = parser.parse_args()

    # Train the model using synthetic data
    model, scaler, poly = train_predictive_model()

    # Save the model to GCS
    save_model_to_gcs(model, scaler, poly, args.model_bucket, args.model_prefix)

    # Process images and save predictions
    process_images_in_bucket(
        args.input_bucket, 
        args.output_bucket, 
        model, 
        scaler, 
        poly, 
        args.slice_number, 
        args.patch_number, 
        args.paddock_number, 
        args.bq_dataset, 
        args.bq_table
    )