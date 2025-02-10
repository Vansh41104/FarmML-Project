from flask import Flask, request, jsonify
import os

# Import your existing processor classes
from latsandlongs import PaddockProcessor
from metadata import ImageMetadataProcessor
from quality import ImageProcessor

app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './agritwin-cv-5867208ae5d4.json'

@app.route('/api/latlon', methods=['POST'])
def process_lat_lon():
    try:
        data = request.get_json()
        required_fields = [
            'bucket_name', 'project_id', 'paddock_dataset_id',
            'paddock_table_id', 'results_dataset_id', 'results_table_id'
        ]

        # Check for missing fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Instantiate PaddockProcessor with data from the request
        processor = PaddockProcessor(
            project_id=data['project_id'],
            bucket_name=data['bucket_name'],
            paddock_dataset_id=data['paddock_dataset_id'],
            paddock_table_id=data['paddock_table_id'],
            results_dataset_id=data['results_dataset_id'],
            results_table_id=data['results_table_id'],
            prefix=data.get('prefix')  # Optional field
        )

        processor.run()
        return jsonify({"status": "success", "message": "Paddock processing completed"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metadata', methods=['POST'])
def process_metadata():
    try:
        data = request.get_json()
        required_fields = ['bucket', 'project', 'dataset', 'table']
        
        # Check for missing required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Initialize processor instance with parameters
        processor = ImageMetadataProcessor(
            bucket_name=data['bucket'],
            project_id=data['project'],
            dataset_id=data['dataset'],
            table_id=data['table'],
            prefix=data.get('prefix', 'Drone/2022-02-19/DCIM/100FPLAN/RGB')
        )
        
        # Run the processor
        processor.process_bucket()
        return jsonify({"status": "success", "message": "Metadata processing completed"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/quality', methods=['POST'])
def process_quality():
    try:
        data = request.get_json()
        required_fields = ['input_bucket', 'bq_dataset', 'bq_table']
        
        # Check for missing required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Initialize processor instance with parameters
        processor = ImageProcessor(
            input_bucket=data['input_bucket'],
            bq_dataset=data['bq_dataset'],
            bq_table=data['bq_table'],
            prefix=data.get('prefix', 'Drone/2022-02-19/DCIM/100FPLAN/RGB'),
            limit=data.get('limit', 13550)
        )
        
        # Run the processor
        processor.process_images()
        return jsonify({"status": "success", "message": "Image quality processing completed"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
