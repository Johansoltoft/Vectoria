# routes/api_routes.py
from flask import Blueprint, request, jsonify
from services.model_service import ModelService
import pandas as pd
import numpy as np

def create_blueprint(model_service: ModelService):
    bp = Blueprint('api', __name__)
    
    @bp.route('/process', methods=['POST'])
    def process_data():
        try:
            file = request.files['file']
            text_column = request.form['text_column']
            
            # Read CSV in chunks
            chunk_size = 1000  # Adjust based on your memory constraints
            chunks = pd.read_csv(file, chunksize=chunk_size)
            
            all_embeddings = []
            all_texts = []
            
            for chunk in chunks:
                # Process each row in the chunk
                chunk['embeddings'] = chunk[text_column].apply(
                    lambda x: model_service.get_embeddings([x])[0] if pd.notna(x) else None
                )
                
                # Filter out rows where embedding failed
                valid_rows = chunk['embeddings'].notna()
                
                # Collect valid embeddings and texts
                all_embeddings.extend(chunk.loc[valid_rows, 'embeddings'].tolist())
                all_texts.extend(chunk.loc[valid_rows, text_column].tolist())
            
            # Convert to numpy array for BERTopic
            embeddings_array = np.vstack(all_embeddings)
            
            # Generate visualization and topics
            plot_data = model_service.create_visualization_and_topics(
                embeddings_array, 
                all_texts
            )
            
            return jsonify({
                'plot_data': plot_data,
                'num_points': len(all_texts)
            })
            
        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            return jsonify({'error': str(e)}), 400
            
    return bp