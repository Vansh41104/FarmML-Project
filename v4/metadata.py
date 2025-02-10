import os
from datetime import datetime
from google.cloud import storage, bigquery
import argparse

class ImageMetadataProcessor:
    def __init__(self, bucket_name, project_id, dataset_id, table_id, prefix=None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.prefix = prefix
        self.storage_client = storage.Client()
        self.bigquery_client = bigquery.Client()

        
    def process_filename(self, filename):
        path_parts = filename.split('/')
        date_part = path_parts[1]  # Assuming format 'YYYY-MM-DD'
        
        # Parse the date_part to a datetime object
        date_time_obj = datetime.strptime(date_part, "%Y-%m-%d")
        date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Split the file name by '_'
        parts = path_parts[-1].split('_')
        
        # Generate id
        image_id = f'{parts[4]}_{parts[5]}'  # e.g., '100073_DJI'
        
        # Parsing numeric values
        latitude = float(parts[7])
        longitude = float(parts[8])
        attitude = float(parts[9].replace('+', ''))
        paddock_number = int(parts[3])
        slice_number = int(parts[10])
        weight = int(parts[11])
        type_index = int(parts[12].split('.')[0])  # Extract only the number part
        
        return {
            'id': image_id,
            'date': date,
            'paddock_number': paddock_number,
            'latitude': latitude,
            'longitude': longitude,
            'attitude': attitude,
            'slice_number': slice_number,
            'weight': weight,
            'type_index': type_index,
            'file_name': filename,
            'stage': None,
            'quality_index': None
        }

    def setup_bigquery_table(self):
        dataset_ref = self.bigquery_client.dataset(self.dataset_id)
        table_ref = dataset_ref.table(self.table_id)
        
        # Define the schema for BigQuery
        schema = [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("date", "TIMESTAMP"),
            bigquery.SchemaField("paddock_number", "STRING"),
            bigquery.SchemaField("latitude", "FLOAT"),
            bigquery.SchemaField("longitude", "FLOAT"),
            bigquery.SchemaField("attitude", "FLOAT"),
            bigquery.SchemaField("slice_number", "INTEGER"),
            bigquery.SchemaField("weight", "FLOAT"),
            bigquery.SchemaField("type_index", "INTEGER"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("stage", "INTEGER"),  # New field
            bigquery.SchemaField("quality_index", "FLOAT")  # New field
        ]
        
        try:
            table = self.bigquery_client.get_table(table_ref)
            print(f"Table '{self.table_id}' already exists.")
        except Exception as e:
            print(f"Table '{self.table_id}' does not exist. Creating it.")
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bigquery_client.create_table(table)
            print(f"Created table '{self.table_id}'.")

        return table

    def process_bucket(self):
        bucket = self.storage_client.get_bucket(self.bucket_name)
        table = self.setup_bigquery_table()
        blobs = bucket.list_blobs(prefix=self.prefix)
        
        rows_to_insert = []
        for blob in blobs:
            print(f"Found file: {blob.name}")
            if blob.name.endswith('.TIF'):
                try:
                    file_info = self.process_filename(blob.name)
                    rows_to_insert.append(file_info)
                    
                    if len(rows_to_insert) >= 100:
                        self.insert_rows_to_bigquery(table, rows_to_insert)
                        rows_to_insert = []

                except Exception as e:
                    print(f"Error processing file '{blob.name}': {e}")

        if rows_to_insert:
            self.insert_rows_to_bigquery(table, rows_to_insert)

        print(f"Processed files from '{self.prefix}' and inserted data into BigQuery table '{self.project_id}.{self.dataset_id}.{self.table_id}'.")

    def insert_rows_to_bigquery(self, table, rows_to_insert):
        errors = self.bigquery_client.insert_rows_json(table, rows_to_insert)
        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
        else:
            print("Rows successfully inserted into BigQuery.")

# if _name_ == "_main_":
#     parser = argparse.ArgumentParser(description='Process image files from GCS and insert metadata into BigQuery.')
#     parser.add_argument('--bucket', required=True, help='GCS bucket name')
#     parser.add_argument('--project', required=True, help='Google Cloud project ID')
#     parser.add_argument('--dataset', required=True, help='BigQuery dataset ID')
#     parser.add_argument('--table', required=True, help='BigQuery table ID')
#     parser.add_argument('--prefix', default='Drone/2022-02-19/DCIM/100FPLAN/RGB', help='Prefix for GCS files')

#     args = parser.parse_args()

#     processor = ImageMetadataProcessor(
#         bucket_name=args.bucket,
#         project_id=args.project,
#         dataset_id=args.dataset,
#         table_id=args.table,
#         prefix=args.prefix
#     )
#     processor.process_bucket()