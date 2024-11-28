# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, AutoModel
# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
# import torch
# import pandas as pd
# import umap.umap_ as umap
# import numpy as np
# import json
# from scipy.stats import gaussian_kde

# app = Flask(__name__)
# CORS(app)

# # Load the model and tokenizer
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# def get_embeddings(texts):
#     # Tokenize and get model outputs
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Use mean pooling to get sentence embeddings
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.numpy()

# def create_topic_visualization(embeddings, texts):
#     # Create UMAP reduction
#     umap_model = umap.UMAP(
#         n_neighbors=min(15, len(texts)-1),
#         min_dist=0.1,
#         n_components=2,
#         metric='cosine',
#         random_state=42
#     )
    
#     # Perform UMAP reduction directly
#     umap_embeddings = umap_model.fit_transform(embeddings)
    
#     # Initialize BERTopic with more permissive parameters
#     vectorizer = CountVectorizer(
#         stop_words="english",
#         min_df=1,
#         max_df=1.0
#     )
    
#     topic_model = BERTopic(
#         embedding_model=None,
#         umap_model=umap_model,
#         vectorizer_model=vectorizer,
#         min_topic_size=2,
#         verbose=True
#     )
    
#     # Fit the model and get topics
#     topics, probs = topic_model.fit_transform(texts, embeddings)
    
#     # Convert to numpy array for easier manipulation
#     topics_array = np.array(topics)
    
#     # Create DataFrame for easier manipulation
#     df = pd.DataFrame({
#         'x': umap_embeddings[:, 0],
#         'y': umap_embeddings[:, 1],
#         'topic': topics,
#         'text': texts
#     })
    
#     # Get topic info and keywords
#     topic_keywords = {}
#     topic_info = topic_model.get_topic_info()
#     unique_topics = sorted(list(set(topics)))
    
#     for topic_id in unique_topics:
#         if topic_id != -1:
#             try:
#                 keywords = [word for word, _ in topic_model.get_topic(topic_id)][:3]
#                 topic_keywords[str(topic_id)] = " | ".join(keywords)
#             except:
#                 topic_keywords[str(topic_id)] = f"Topic {topic_id}"
    
#     # Calculate topic centers and sizes
#     topic_centers = {}
#     for topic_id in unique_topics:
#         if topic_id != -1:
#             topic_docs = df[df['topic'] == topic_id]
#             if not topic_docs.empty:
#                 # Calculate center
#                 center_x = float(topic_docs['x'].mean())
#                 center_y = float(topic_docs['y'].mean())
                
#                 # Calculate size (number of documents)
#                 size = len(topic_docs)
                
#                 # Calculate density for this topic
#                 try:
#                     xy = np.vstack([topic_docs['x'], topic_docs['y']])
#                     z = gaussian_kde(xy)(xy)
#                     density = float(z.mean())
#                 except:
#                     density = 1.0
                
#                 topic_centers[str(topic_id)] = {
#                     'x': center_x,
#                     'y': center_y,
#                     'keywords': topic_keywords.get(str(topic_id), f"Topic {topic_id}"),
#                     'size': size,
#                     'density': density
#                 }
    
#     # Prepare visualization data
#     plot_data = {
#         'x': df['x'].tolist(),
#         'y': df['y'].tolist(),
#         'texts': texts,
#         'topics': topic_model.topics_,
#         'topic_centers': topic_centers,
#         'topic_keywords': topic_keywords,
#         'unique_topics': [t for t in unique_topics if t != -1],
#         'type': 'scatter'
#     }
    
#     # Add document counts per topic
#     plot_data['topic_sizes'] = {
#         str(topic): int(sum(topics_array == topic))
#         for topic in unique_topics if topic != -1
#     }
    
#     # Calculate global ranges for better visualization
#     plot_data['x_range'] = [float(df['x'].min()), float(df['x'].max())]
#     plot_data['y_range'] = [float(df['y'].min()), float(df['y'].max())]
    
#     return plot_data

# @app.route('/process', methods=['POST'])
# def process_data():
#     try:
#         # Get file and column name from request
#         file = request.files['file']
#         text_column = request.form['text_column']
        
#         # Read CSV file
#         df = pd.read_csv(file)
        
#         # Get texts from specified column
#         texts = df[text_column].fillna('').tolist()
        
#         if len(texts) == 0:
#             return jsonify({'error': 'No texts found in the specified column'}), 400
        
#         # Generate embeddings
#         embeddings = get_embeddings(texts)
        
#         # Create visualization data with topics
#         plot_data = create_topic_visualization(embeddings, texts)
        
#         return jsonify({
#             'plot_data': plot_data,
#             'num_points': len(texts)
#         })
#     except Exception as e:
#         print(f"Error in process_data: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, AutoModel
# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
# import torch
# import pandas as pd
# import umap.umap_ as umap
# import numpy as np
# import json
# from scipy.stats import gaussian_kde

# app = Flask(__name__)
# CORS(app)

# # Load the model and tokenizer
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# def get_embeddings(texts):
#     # Tokenize and get model outputs
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Use mean pooling to get sentence embeddings
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.numpy()

# def create_topic_visualization(embeddings, texts):
#     # UMAP reduction
#     umap_model = umap.UMAP(
#         n_neighbors=min(15, len(texts)-1),
#         min_dist=0.1,
#         n_components=2,
#         metric='cosine',
#         random_state=42
#     )
    
#     # Get 2D coordinates
#     umap_embeddings = umap_model.fit_transform(embeddings)
    
#     # Topic modeling
#     vectorizer = CountVectorizer(stop_words="english")
#     topic_model = BERTopic(
#         embedding_model=None,
#         umap_model=umap_model,
#         vectorizer_model=vectorizer,
#         min_topic_size=2
#     )
    
#     # Get topics
#     topics, _ = topic_model.fit_transform(texts, embeddings)
    
#     # Professional color palette
#     colors = [
#         '#4363d8', '#42d4f4', '#469990', '#dcbeff', '#9A6324',
#         '#800000', '#808000', '#3cb44b', '#000075', '#f58231'
#     ]
    
#     # Organize data in dataframe
#     df = pd.DataFrame({
#         'x': umap_embeddings[:, 0],
#         'y': umap_embeddings[:, 1],
#         'topic': topics,
#         'text': texts
#     })
    
#     # Process topic information
#     topic_centers = {}
#     unique_topics = sorted(list(set(topics)))
    
#     for topic_id in unique_topics:
#         if topic_id != -1:  # Skip outlier topic if any
#             topic_docs = df[df['topic'] == topic_id]
#             if not topic_docs.empty:
#                 # Get topic keywords
#                 keywords = topic_model.get_topic(topic_id)
#                 topic_name = " | ".join([word for word, _ in keywords[:3]])
                
#                 # Calculate center
#                 center_x = float(topic_docs['x'].mean())
#                 center_y = float(topic_docs['y'].mean())
                
#                 # Generate density map
#                 xx, yy, density = generate_density_heatmap(
#                     topic_docs['x'].values,
#                     topic_docs['y'].values
#                 )
                
#                 topic_centers[str(topic_id)] = {
#                     'x': center_x,
#                     'y': center_y,
#                     'keywords': topic_name,
#                     'size': len(topic_docs),
#                     'color': colors[topic_id % len(colors)],
#                     'color_transparent': colors[topic_id % len(colors)].replace(
#                         'rgb', 'rgba').replace(')', ',0.2)'
#                     ),
#                     'density_map': {
#                         'x': xx[0].tolist(),
#                         'y': yy[:, 0].tolist(),
#                         'z': density.tolist()
#                     }
#                 }
    
#     return {
#         'x': df['x'].tolist(),
#         'y': df['y'].tolist(),
#         'texts': texts,
#         'topics': topic_model.topics_,
#         'topic_centers': topic_centers,
#         'x_range': [float(df['x'].min() - 0.5), float(df['x'].max() + 0.5)],
#         'y_range': [float(df['y'].min() - 0.5), float(df['y'].max() + 0.5)]
#     }

# def generate_density_heatmap(x, y, resolution=100):
#     """Simple density heatmap generation."""
#     x_min, x_max = min(x), max(x)
#     y_min, y_max = min(y), max(y)
    
#     # Create grid
#     x_range = np.linspace(x_min - 0.5, x_max + 0.5, resolution)
#     y_range = np.linspace(y_min - 0.5, y_max + 0.5, resolution)
#     xx, yy = np.meshgrid(x_range, y_range)
    
#     # Generate density
#     positions = np.vstack([xx.ravel(), yy.ravel()])
#     values = np.vstack([x, y])
#     kernel = gaussian_kde(values)
#     z = np.reshape(kernel(positions).T, xx.shape)
    
#     return xx, yy, z

# @app.route('/process', methods=['POST'])
# def process_data():
#     try:
#         # Get file and column name from request
#         file = request.files['file']
#         text_column = request.form['text_column']
        
#         # Read CSV file
#         df = pd.read_csv(file)
        
#         # Get texts from specified column
#         texts = df[text_column].fillna('').tolist()
        
#         if len(texts) == 0:
#             return jsonify({'error': 'No texts found in the specified column'}), 400
        
#         # Generate embeddings
#         embeddings = get_embeddings(texts)
        
#         # Create visualization data with topics
#         plot_data = create_topic_visualization(embeddings, texts)
        
#         return jsonify({
#             'plot_data': plot_data,
#             'num_points': len(texts)
#         })
#     except Exception as e:
#         print(f"Error in process_data: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# app.py
from flask import Flask
from flask_cors import CORS
from routes import api_routes
from services.model_service import ModelService


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize services
    model_service = ModelService()
    
    # Register blueprints
    app.register_blueprint(api_routes.create_blueprint(model_service))
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)