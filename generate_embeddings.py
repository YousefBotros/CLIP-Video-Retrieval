"""
Generate CLIP embeddings for video frames
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import pickle
from tqdm import tqdm

class CLIPEmbeddingGenerator:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize CLIP model for generating embeddings
        
        Args:
            model_name: CLIP model variant
            device: 'cuda' or 'cpu'
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def get_image_embedding(self, image_path):
        """Generate embedding for a single image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        # Normalize embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def get_text_embedding(self, text):
        """Generate embedding for a text query"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        
        # Normalize embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def generate_all_embeddings(self, frame_dir, output_file="frame_embeddings.pkl"):
        """
        Generate embeddings for all frames in a directory
        
        Args:
            frame_dir: Directory containing frame images
            output_file: Path to save embeddings
        """
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if not frame_files:
            raise ValueError(f"No image files found in {frame_dir}")
        
        print(f"Found {len(frame_files)} frames to process")
        
        embeddings = []
        frame_paths = []
        
        for frame_file in tqdm(frame_files, desc="Generating embeddings"):
            frame_path = os.path.join(frame_dir, frame_file)
            embedding = self.get_image_embedding(frame_path)
            embeddings.append(embedding)
            frame_paths.append(frame_path)
        
        # Save embeddings
        data = {
            'frame_paths': frame_paths,
            'embeddings': np.array(embeddings),
            'num_frames': len(frame_paths)
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved {len(embeddings)} embeddings to {output_file}")
        return data
    
    def load_embeddings(self, embedding_file):
        """Load pre-computed embeddings"""
        with open(embedding_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded {data['num_frames']} embeddings from {embedding_file}")
        return data

def create_faiss_index(embeddings):
    """Create FAISS index for fast similarity search"""
    import faiss
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
    index.add(embeddings.astype('float32'))
    
    return index

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for video frames")
    parser.add_argument("frame_dir", help="Directory containing extracted frames")
    parser.add_argument("--output", default="frame_embeddings.pkl", help="Output file for embeddings")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    
    args = parser.parse_args()
    
    # Generate embeddings
    generator = CLIPEmbeddingGenerator(model_name=args.model)
    embeddings_data = generator.generate_all_embeddings(args.frame_dir, args.output)
    
    # Create FAISS index
    import faiss
    index = create_faiss_index(embeddings_data['embeddings'])
    
    # Save FAISS index
    faiss.write_index(index, "frame_index.faiss")
    print("✅ FAISS index saved to frame_index.faiss")
