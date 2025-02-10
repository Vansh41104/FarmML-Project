import argparse
import os
from datetime import datetime
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from shapely.geometry import Point, Polygon

class PaddockProcessor:
    def __init__(self, project_id, bucket_name, paddock_dataset_id, paddock_table_id, results_dataset_id, results_table_id, prefix=None):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.paddock_dataset_id = paddock_dataset_id
        self.paddock_table_id = paddock_table_id
        self.results_dataset_id = results_dataset_id
        self.results_table_id = results_table_id
        self.prefix = prefix
        self.paddocks = self.read_paddock_bigquery()
    
    def process_filename(self, filename):
        path_parts = filename.split('/')
        date_part = path_parts[1]
        parts = path_parts[-1].split('_')
        latitude = float(parts[7])
        longitude = float(parts[8])
        paddock_number = int(parts[3])
        return {
            'image_name': filename,
            'latitude': latitude,
            'longitude': longitude,
            'paddock_number': paddock_number
        }
    
    def read_paddock_bigquery(self):
        client = bigquery.Client(project=self.project_id)
        query = f"""
        SELECT Paddock, Latitude, Longitude
        FROM {self.project_id}.{self.paddock_dataset_id}.{self.paddock_table_id}
        ORDER BY Paddock, 
        CASE 
            WHEN SAFE_CAST(SUBSTR(Paddock, -2) AS INT64) IS NOT NULL 
            THEN SAFE_CAST(SUBSTR(Paddock, -2) AS INT64) 
            ELSE 0 
        END
        """
        query_job = client.query(query)
        results = query_job.result()

        paddocks = {}
        current_paddock = None
        corners = []

        for row in results:
            paddock_no = row['Paddock'][:-3]
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            
            if paddock_no != current_paddock:
                if current_paddock is not None:
                    paddocks[current_paddock] = Polygon(corners)
                current_paddock = paddock_no
                corners = []
            
            corners.append((lon, lat))
            
            if row['Paddock'].endswith('_c5'):
                paddocks[current_paddock] = Polygon(corners)
        
        if current_paddock is not None and current_paddock not in paddocks:
            paddocks[current_paddock] = Polygon(corners)

        print(f"Total paddocks loaded from BigQuery: {len(paddocks)}")
        return paddocks
    
    def assign_paddock(self, point):
        for paddock_id, polygon in self.paddocks.items():
            if polygon.contains(point):
                return paddock_id
        return "None"
    
    def process_images(self):
        credentials = service_account.Credentials.from_service_account_file('agritwin-cv-5867208ae5d4.json')
        storage_client = storage.Client(credentials=credentials, project='agritwin-cv')
        bucket = storage_client.bucket(self.bucket_name)
        
        results = []
        total_files = 0
        tif_files = 0
        processed_files = 0
        skipped_files = 0
        error_files = 0

        for blob in bucket.list_blobs(prefix=self.prefix):
            total_files += 1
            if blob.name.lower().endswith('.tif'):
                tif_files += 1
                try:
                    print(f"Processing file: {blob.name}")
                    
                    file_info = self.process_filename(blob.name)
                    lat, lon = file_info['latitude'], file_info['longitude']
                    point = Point(lon, lat)
                    predicted_paddock_id = self.assign_paddock(point)
                    
                    result = {
                        'image_name': file_info['image_name'],
                        'center_point': f"POINT({lon} {lat})",
                        'paddock_no': predicted_paddock_id
                    }
                    
                    results.append(result)
                    processed_files += 1
                    
                except Exception as e:
                    print(f"Error processing {blob.name}: {str(e)}")
                    error_files += 1
            else:
                print(f"Skipping non-TIF file: {blob.name}")
                skipped_files += 1
        
        print(f"Total files in bucket: {total_files}")
        print(f"TIF files found: {tif_files}")
        print(f"Successfully processed files: {processed_files}")
        print(f"Skipped files: {skipped_files}")
        print(f"Files with errors: {error_files}")
        
        return results
    
    def write_to_bigquery(self, data):
        client = bigquery.Client(project=self.project_id)
        table_ref = client.dataset(self.results_dataset_id).table(self.results_table_id)
        
        schema = [
            bigquery.SchemaField("image_name", "STRING"),
            bigquery.SchemaField("center_point", "GEOGRAPHY"),
            bigquery.SchemaField("paddock_no", "STRING"),
        ]
        
        try:
            client.get_table(table_ref)
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
        
        if not data:
            print("No data to insert. Skipping BigQuery insertion.")
            return
        
        errors = client.insert_rows_json(table_ref, data)
        if errors:
            print("Errors occurred while inserting rows:")
            for error in errors:
                print(error)
        else:
            print(f"Data successfully uploaded to BigQuery table {self.project_id}.{self.results_dataset_id}.{self.results_table_id}")
    
    def run(self):
        results = self.process_images()
        self.write_to_bigquery(results)

# if _name_ == "_main_":
#     parser = argparse.ArgumentParser(description="Process images from GCS, extract metadata, predict paddocks, and write results to BigQuery")
#     parser.add_argument("--bucket_name", required=True, help="GCS bucket name containing images")
#     parser.add_argument("--project_id", required=True, help="Google Cloud project ID")
#     parser.add_argument("--paddock_dataset_id", required=True, help="BigQuery dataset ID for paddock data")
#     parser.add_argument("--paddock_table_id", required=True, help="BigQuery table ID for paddock data")
#     parser.add_argument("--results_dataset_id", required=True, help="BigQuery dataset ID for results")
#     parser.add_argument("--results_table_id", required=True, help="BigQuery table ID for results")
#     parser.add_argument("--prefix", required=False, help="Optional prefix to filter the GCS files by name")
    
#     args = parser.parse_args()
    
#     processor = PaddockProcessor(
#         args.project_id, 
#         args.bucket_name, 
#         args.paddock_dataset_id, 
#         args.paddock_table_id, 
#         args.results_dataset_id, 
#         args.results_table_id, 
#         args.prefix
#     )
#     processor.run()