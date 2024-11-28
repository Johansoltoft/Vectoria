# services/model_service.py
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import torch
import pandas as pd
import umap.umap_ as umap
import numpy as np
from scipy.stats import gaussian_kde

class ModelService:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.colors = [
            '#4363d8', '#42d4f4', '#469990', '#dcbeff', '#9A6324',
            '#800000', '#808000', '#3cb44b', '#000075', '#f58231'
        ]
        
        # Initialize UMAP models
            ### for BERTopic -----------------------------
        self.umap_model = umap.UMAP(
            n_neighbors=20,
            n_components=50,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
            ### for 2D embeddings -----------------------------
        self.umap_model_2d = umap.UMAP(
            n_neighbors=20,
            n_components=2,
            min_dist=0.15,
            metric='cosine',
            random_state=42
        )
        
        # Initialize HDBSCAN - CLUSTERING
            ### for BERTopic -----------------------------
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
        )
        
        # Initialize other BERTopic components
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.representation_models = {
            "keybert": KeyBERTInspired(top_n_words=10)
        }

    def get_embeddings(self, texts):
        """
        Get embeddings for texts. Can handle single text or list of texts.
        Returns numpy array of embeddings.
        """
        # Handle input type (single string vs list)
        if isinstance(texts, str):
            texts = [texts]
        
        # Process texts
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        
        # Move to CPU to free GPU memory if available
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu()  # Move to CPU
            
        # Clean up to free memory
        del outputs
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return embeddings.numpy()

    def generate_density_heatmap(self, x, y, resolution=100):
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        
        x_range = np.linspace(x_min - 0.5, x_max + 0.5, resolution)
        y_range = np.linspace(y_min - 0.5, y_max + 0.5, resolution)
        xx, yy = np.meshgrid(x_range, y_range)
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        z = np.reshape(kernel(positions).T, xx.shape)
        
        return xx, yy, z

    def create_visualization_and_topics(self, embeddings, texts):
        # Get 2D coordinates for visualization
        viz_embeddings = self.umap_model_2d.fit_transform(embeddings)
        
        # Initialize and fit BERTopic
        topic_model = BERTopic(
            embedding_model=self.model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            verbose=True,
            top_n_words=10,
            representation_model=self.representation_models,
            calculate_probabilities=True
        )
        
        # Fit topic model
        topics, probs = topic_model.fit_transform(texts, embeddings)
        
        # Convert to a DataFrame
        df_topics = pd.DataFrame(topic_model.get_topic_info())
        
        # Create DataFrame with 2D visualization coordinates
        df = pd.DataFrame({
            'x': viz_embeddings[:, 0],
            'y': viz_embeddings[:, 1],
            'topic': topics,  # Use topics directly instead of df_topics['Topic']
            'text': texts,
            'probability': probs.max(axis=1)
        })
        
        topic_centers = {}
        # Use df_topics to iterate through topics
        for _, topic_row in df_topics.iterrows():
            topic_id = topic_row['Topic']
            if topic_id != -1:  # Skip outlier topic if any
                topic_docs = df[df['topic'] == topic_id]
                if not topic_docs.empty:
                    # Use the Name column from df_topics if you want the auto-generated name
                    # or use keybert column for keyword-based representation
                    if isinstance(topic_row['keybert'], list):
                        # If keybert column contains list of tuples/lists
                        keywords = topic_row['keybert'][:4]
                        topic_name = " | ".join([word[0] if isinstance(word, (tuple, list)) else str(word) for word in keywords])
                    else:
                        # Fallback to Name column
                        topic_name = topic_row['Name']
                    
                    # Calculate center in visualization space
                    center_x = float(topic_docs['x'].mean())
                    center_y = float(topic_docs['y'].mean())
                    
                    # Generate density map
                    xx, yy, density = self.generate_density_heatmap(
                        topic_docs['x'].values,
                        topic_docs['y'].values
                    )
                    
                    # Get top documents for this topic
                    topic_docs_sorted = topic_docs.sort_values('probability', ascending=False)
                    top_docs = topic_docs_sorted['text'].head(5).tolist()
                    
                    topic_centers[str(topic_id)] = {
                        'x': center_x,
                        'y': center_y,
                        'keywords': topic_name,
                        'size': topic_row['Count'],  # Use Count from df_topics
                        'color': self.colors[topic_id % len(self.colors)],
                        'color_transparent': self.colors[topic_id % len(self.colors)].replace(
                            'rgb', 'rgba').replace(')', ',0.2)'
                        ),
                        'density_map': {
                            'x': xx[0].tolist(),
                            'y': yy[:, 0].tolist(),
                            'z': density.tolist()
                        },
                        'top_documents': top_docs,
                        'coherence_score': float(topic_docs['probability'].mean())
                    }
        
        return {
            'x': df['x'].tolist(),
            'y': df['y'].tolist(),
            'texts': texts,
            'topics': topics,  # Use the topics directly
            'probabilities': probs.tolist(),
            'topic_centers': topic_centers,
            'x_range': [float(df['x'].min() - 0.5), float(df['x'].max() + 0.5)],
            'y_range': [float(df['y'].min() - 0.5), float(df['y'].max() + 0.5)]
        }

    def process_texts(self, texts):
        embeddings = self.get_embeddings(texts)
        return self.create_visualization_and_topics(embeddings, texts)