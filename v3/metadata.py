import os
from datetime import datetime
from google.cloud import storage
from google.cloud import bigquery
import argparse

def process_filename(filename):
    path_parts = filename.split('/')
    date_part = path_parts[1]  # Assuming format 'YYYY-MM-DD'
    
    # Parse the date_part to a datetime object
    date_time_obj = datetime.strptime(date_part, "%Y-%m-%d")
    
    # Use 'date' instead of 'date_time'
    date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Split the file name by '_'
    parts = path_parts[-1].split('_')
    
    # Generate id (combination of parts or could be auto-incremented based on your logic)
    image_id = f'{parts[4]}_{parts[5]}'  # e.g., '100073_DJI'
    
    # Parsing numeric values (ignoring DJI part)
    latitude = float(parts[7])
    longitude = float(parts[8])
    attitude = float(parts[9].replace('+', ''))
    paddock_number = int(parts[3])
    slice_number = int(parts[10])
    weight = int(parts[11])
    type_index = int(parts[12].split('.')[0])  # Extract only the number part
    
    return {
        'id': image_id,  # Use id for primary key
        'date': date,  # Insert into 'date' field
        'paddock_number': paddock_number,
        'latitude': latitude,
        'longitude': longitude,
        'attitude': attitude,  # Insert 'altitude'
        'slice_number': slice_number,
        'weight': weight,
        'type_index': type_index,
        'file_name': filename  # Use file_name instead of filename
    }

def process_bucket(bucket_name, project_id, dataset_id, table_id, prefix):
    # Initialize clients
    storage_client = storage.Client()
    bigquery_client = bigquery.Client()

    # Get the bucket
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        return

    # Prepare BigQuery table reference
    dataset_ref = bigquery_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

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
    ]

    # Create the table if it doesn't exist
    try:
        table = bigquery_client.get_table(table_ref)
        print(f"Table '{table_id}' already exists.")
    except Exception as e:
        print(f"Error: {e}")
        # Table doesn't exist, create it
        table = bigquery.Table(table_ref, schema=schema)
        table = bigquery_client.create_table(table)
        print(f"Created table '{table_id}'.")

    # List blobs in the bucket recursively with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    rows_to_insert = []

    for blob in blobs:
        print(f"Found file: {blob.name}")
        # Only process .TIF files
        if blob.name.endswith('.TIF'):
            try:
                # Process the filename to extract metadata
                file_info = process_filename(blob.name)
                rows_to_insert.append(file_info)

                # Insert rows into BigQuery in batches of 100
                if len(rows_to_insert) >= 100:
                    errors = bigquery_client.insert_rows_json(table, rows_to_insert)
                    if errors:
                        print(f"Encountered errors while inserting rows: {errors}")
                    rows_to_insert = []

            except Exception as e:
                print(f"Error processing file '{blob.name}': {e}")

    # Insert any remaining rows
    if rows_to_insert:
        errors = bigquery_client.insert_rows_json(table, rows_to_insert)
        if errors:
            print(f"Encountered errors while inserting rows: {errors}")

    print(f"Processed files from '{prefix}' and inserted data into BigQuery table '{project_id}.{dataset_id}.{table_id}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image files from GCS and insert metadata into BigQuery.')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--project', required=True, help='Google Cloud project ID')
    parser.add_argument('--dataset', required=True, help='BigQuery dataset ID')
    parser.add_argument('--table', required=True, help='BigQuery table ID')
    parser.add_argument('--prefix', default='Drone/2022-02-19/DCIM/100FPLAN/RGB', help='Prefix for GCS files')

    args = parser.parse_args()

    process_bucket(args.bucket, args.project, args.dataset, args.table, args.prefix)
