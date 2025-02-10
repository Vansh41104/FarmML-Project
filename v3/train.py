import os
import argparse
import tensorflow as tf
from google.cloud import storage
from google.cloud import bigquery
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
import csv
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
from skimage.feature import greycomatrix, greycoprops

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './agritwin-cv-5867208ae5d4.json'

# Initialize clients
storage_client = storage.Client()
bigquery_client = bigquery.Client()

def download_image_from_bucket(bucket_name, image_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(image_name)
    blob.download_to_filename(destination_file_name)

def calculate_ndvi(image):
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    ndvi = (green - red) / (green + red + 1e-7)
    return ndvi

def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    glcm = greycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    contrast = greycoprops(glcm, 'contrast').flatten()
    dissimilarity = greycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_image_features(image_path):
    with rasterio.open(image_path) as dataset:
        data = dataset.read(
            out_shape=(3, 224, 224),
            resampling=Resampling.bilinear
        )
        data = data.transpose(1, 2, 0)
        data = data.astype(np.float32) / 255.0
        
        if data.shape[2] == 1:
            data = np.repeat(data, 3, axis=2)
        elif data.shape[2] > 3:
            data = data[:, :, :3]

    # Extract basic statistics
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    min_val = np.min(data, axis=(0, 1))
    max_val = np.max(data, axis=(0, 1))

    # Calculate NDVI
    red = data[:, :, 0]
    green = data[:, :, 1]
    ndvi = (green - red) / (green + red + 1e-7)
    ndvi_mean = np.mean(ndvi)
    ndvi_std = np.std(ndvi)

    # Extract GLCM features
    glcm = greycomatrix(np.mean(data, axis=2).astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    features = np.concatenate([
        mean, std, min_val, max_val,
        [ndvi_mean, ndvi_std, contrast, dissimilarity, homogeneity, energy, correlation]
    ])

    return features

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    model = create_model(input_shape=(X_train_scaled.shape[1],))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    y_pred_scaled = model.predict(X_val_scaled).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Plot residuals
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.savefig('residuals_plot.png')
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return model, mae, r2, scaler_X, scaler_y

def main():
    parser = argparse.ArgumentParser(description="Train and validate an optimized grass weight prediction model.")
    parser.add_argument("--input_bucket", type=str, required=True, help="GCS bucket with input images.")
    parser.add_argument("--input_prefix", type=str, required=True, help="Prefix for input images in GCS.")
    parser.add_argument("--bq_project", type=str, required=True, help="BigQuery project name.")
    parser.add_argument("--bq_dataset", type=str, required=True, help="BigQuery dataset name.")
    parser.add_argument("--bq_table_input", type=str, required=True, help="BigQuery table for input data.")
    parser.add_argument("--model_output", type=str, required=True, help="Output file for saving trained model.")
    parser.add_argument("--results_output", type=str, required=True, help="Output CSV file for validation results.")
    parser.add_argument("--plot_output", type=str, required=True, help="Output file for results plot.")

    args = parser.parse_args()

    bucket = storage_client.bucket(args.input_bucket)
    blobs = bucket.list_blobs(prefix=args.input_prefix)
    image_names = [blob.name for blob in blobs if blob.name.lower().endswith('.tif')]
    image_names = image_names[:500]  # Limit to 500 images as before

    with tempfile.TemporaryDirectory() as temp_dir:
        X = []
        for image_name in tqdm(image_names, desc="Extracting image features"):
            local_path = os.path.join(temp_dir, os.path.basename(image_name))
            download_image_from_bucket(args.input_bucket, image_name, local_path)
            feature = extract_image_features(local_path)
            X.append(feature)
            os.remove(local_path)

        X = np.array(X)

    print("Fetching data from BigQuery...")
    bq_data = get_data_from_bigquery(args.bq_project, args.bq_dataset, args.bq_table_input)
    bq_data = bq_data[:500]
    y = np.array([row['weight'] for row in bq_data])

    print("Training and evaluating model...")
    model, mae, r2, scaler_X, scaler_y = train_and_evaluate_model(X, y)

    print(f"Saving model to {args.model_output}...")
    model.save(args.model_output)

    print("Making predictions...")
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    print(f"Saving results to {args.results_output}...")
    with open(args.results_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'True Weight', 'Predicted Weight'])
        for image, true, pred in zip(image_names, y, y_pred):
            writer.writerow([image, true, pred])

    print(f"Generating plot and saving to {args.plot_output}...")
    plot_results(y, y_pred, args.plot_output)

    print(f"Process completed. Model saved to {args.model_output}")
    print(f"Results saved to {args.results_output}")
    print(f"Plot saved to {args.plot_output}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Accuracy (R-squared): {r2 * 100:.2f}%")

if __name__ == "__main__":
    main()