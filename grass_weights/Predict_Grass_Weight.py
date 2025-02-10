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
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    red_index = np.mean(R) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    green_index = np.mean(G) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    blue_index = np.mean(B) / (np.mean(R) + np.mean(G) + np.mean(B) + 1e-7)
    
    return {
        'Mean_Intensity': mean_intensity,
        'Variance': variance,
        'Mean_Edges': mean_edges,
        'Mean_Laplacian': mean_laplacian,
        'LBP_1': lbp_hist[0],
        'LBP_2': lbp_hist[1],
        'LBP_3': lbp_hist[2],
        'LBP_4': lbp_hist[3],
        'LBP_5': lbp_hist[4],
        'Entropy': entropy,
        'Red_Index': red_index,
        'Green_Index': green_index,
        'Blue_Index': blue_index
    }

# The rest of the code remains the same

def generate_synthetic_data(n_samples):
    np.random.seed(42)
    synthetic_data = []
    
    for _ in tqdm(range(n_samples), desc="Generating synthetic data"):
        mean_intensity = np.random.uniform(0, 255)
        variance = np.random.uniform(0, 10000)
        mean_edges = np.random.uniform(0, 100)
        mean_laplacian = np.random.uniform(0, 100)
        lbp_1 = np.random.uniform(0, 1)
        lbp_2 = np.random.uniform(0, 1)
        lbp_3 = np.random.uniform(0, 1)
        lbp_4 = np.random.uniform(0, 1)
        lbp_5 = np.random.uniform(0, 1)
        entropy = np.random.uniform(0, 8)
        red_index = np.random.uniform(0, 1)
        green_index = np.random.uniform(0, 1)
        blue_index = np.random.uniform(0, 1)
        
        base_weight = 1 + (variance / 10000)
        noise = np.random.normal(0, 0.1)
        grass_weight = max(1, min(2, base_weight + noise))
        
        synthetic_data.append({
            'Mean_Intensity': mean_intensity,
            'Variance': variance,
            'Mean_Edges': mean_edges,
            'Mean_Laplacian': mean_laplacian,
            'LBP_1': lbp_1,
            'LBP_2': lbp_2,
            'LBP_3': lbp_3,
            'LBP_4': lbp_4,
            'LBP_5': lbp_5,
            'Entropy': entropy,
            'Red_Index': red_index,
            'Green_Index': green_index,
            'Blue_Index': blue_index,
            'Grass_Weight': grass_weight
        })
        
    return pd.DataFrame(synthetic_data)

def train_predictive_model():
    df_synthetic = generate_synthetic_data(n_samples=1000)
    X = df_synthetic.drop('Grass_Weight', axis=1)
    y = df_synthetic['Grass_Weight']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.05],
        'max_depth': [4, 5],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'reg_alpha': [0.1, 0.01],
        'reg_lambda': [1, 1.5]
    }
    
    model = XGBRegressor(tree_method='hist', device='gpu', objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=2, n_jobs=1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test R-squared Score: {r2}")
    
    return model, scaler, poly

def process_image_file(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        
        img = np.uint8(img)
        
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def calculate_grass_weights_with_model(image_parts_dir, model, scaler, poly):
    if not os.path.exists(image_parts_dir):
        print(f"Directory does not exist: {image_parts_dir}")
        return pd.DataFrame()
    
    image_data = []
    image_files = sorted([f for f in os.listdir(image_parts_dir) if f.lower().endswith('.png')])
    
    print(f"Found {len(image_files)} .png files in {image_parts_dir}")
    
    if not image_files:
        print(f"No .png files found in directory: {image_parts_dir}")
        return pd.DataFrame()
    
    for img_file in tqdm(image_files, desc="Processing images"):
        part_path = os.path.join(image_parts_dir, img_file)
        
        img = process_image_file(part_path)
        
        if img is None:
            continue
        
        features = extract_image_features(img)
        
        feature_values = [
            features['Mean_Intensity'], features['Variance'], features['Mean_Edges'],
            features['Mean_Laplacian'], features['LBP_1'], features['LBP_2'],
            features['LBP_3'], features['LBP_4'], features['LBP_5'],
            features['Entropy'],
            features['Red_Index'], features['Green_Index'], features['Blue_Index']
        ]
        
        feature_values = [val.item() if isinstance(val, np.ndarray) else val for val in feature_values]

        try:
            X = np.array([feature_values])
            X_poly = poly.transform(X)
            X_scaled = scaler.transform(X_poly)
            
            predicted_grass_weight = model.predict(X_scaled)[0]
            features['Image'] = img_file
            features['Predicted_Grass_Weight'] = predicted_grass_weight
            
            image_data.append(features)
        except ValueError as ve:
            print(f"Error processing {img_file}: {ve}")
    
    if image_data:
        return pd.DataFrame(image_data)
    else:
        print("No valid images processed.")
        return pd.DataFrame()

def predict_grass_weights_and_save(image_parts_dir, model, scaler, poly, output_csv_path):
    df_features = calculate_grass_weights_with_model(image_parts_dir, model, scaler, poly)
    
    if not df_features.empty:
        df_features.to_csv(output_csv_path, index=False)
        print(f"Grass weights saved to {output_csv_path}")
    else:
        print("No valid images to save.")

if __name__ == "__main__":
    model, scaler, poly = train_predictive_model()
    
    image_parts_dir = "D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_PNG"  # Update this to your PNG images directory
    output_csv_path = "D:/Project Caldermeade/NJR/Training/AG/2022-01-24/DCIM/104FPLAN_20/BG_grass_weights_predictionsss.csv"  # Update this to your desired output path
    
    predict_grass_weights_and_save(image_parts_dir, model, scaler, poly, output_csv_path)

  