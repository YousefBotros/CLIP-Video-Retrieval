"""
Search video frames using text queries
"""

import numpy as np
import pickle
import faiss
from generate_embeddings import CLIPEmbeddingGenerator

class VideoSearcher:
    def __init__(self, embedding_file="frame_embeddings.pkl", index_file="frame_index.faiss"):
        """
        Initialize video searcher with pre-computed embeddings
        
        Args:
            embedding_file: Path to embeddings pickle file
            index_file: Path to FAISS index file
        """
        # Load embeddings
        with open(embedding_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.frame_paths = self.data['frame_paths']
        self.embeddings = self.data['embeddings']
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Initialize CLIP for text embeddings
        self.clip = CLIPEmbeddingGenerator()
        
        print(f"✅ Loaded {len(self.frame_paths)} frames")
    
    def search(self, query, top_k=5):
        """
        Search for frames matching text query
        
        Args:
            query: Text query string
            top_k: Number of top results to return
        
        Returns:
            List of (frame_path, similarity_score) tuples
        """
        # Get text embedding
        text_embedding = self.clip.get_text_embedding(query)
        text_embedding = text_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        scores, indices = self.index.search(text_embedding, top_k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append({
                    'frame_path': self.frame_paths[idx],
                    'frame_index': int(idx),
                    'similarity_score': float(scores[0][i])
                })
        
        return results
    
    def search_and_display(self, query, top_k=5):
        """Search and print results"""
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        results = self.search(query, top_k)
        
        for i, result in enumerate(results):
            print(f"{i+1}. Frame {result['frame_index']} - Score: {result['similarity_score']:.4f}")
            print(f"   Path: {result['frame_path']}\n")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search video frames")
    parser.add_argument("query", nargs="?", default=None, help="Text query to search")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    
    args = parser.parse_args()
    
    searcher = VideoSearcher()
    
    if args.interactive:
        print("\n" + "="*50)
        print("Video Frame Search - Interactive Mode")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            query = input("\n🔍 Enter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if query:
                searcher.search_and_display(query, args.top_k)
    
    elif args.query:
        searcher.search_and_display(args.query, args.top_k)
    
    else:
        print("Please provide a query or use --interactive mode")
