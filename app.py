"""
CLIP Video Retrieval - Gradio Web Application
Search video frames using natural language queries
"""

import gradio as gr
import cv2
import os
from PIL import Image
import tempfile
from search_video import VideoSearcher
from extract_frames import extract_frames, get_video_info

# Global searcher variable
searcher = None

def process_video(video_file, frame_interval=30):
    """Upload and process video, extract frames, generate embeddings"""
    global searcher
    
    if video_file is None:
        return None, "⚠️ Please upload a video file", []
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    frame_dir = os.path.join(temp_dir, "frames")
    
    # Extract frames
    try:
        info = get_video_info(video_file)
        extract_frames(video_file, frame_dir, frame_interval)
        
        # Generate embeddings
        from generate_embeddings import CLIPEmbeddingGenerator
        generator = CLIPEmbeddingGenerator()
        embedding_file = os.path.join(temp_dir, "embeddings.pkl")
        index_file = os.path.join(temp_dir, "index.faiss")
        
        embeddings_data = generator.generate_all_embeddings(frame_dir, embedding_file)
        
        # Create FAISS index
        import faiss
        dimension = embeddings_data['embeddings'].shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_data['embeddings'].astype('float32'))
        faiss.write_index(index, index_file)
        
        # Initialize searcher
        searcher = VideoSearcher(embedding_file, index_file)
        
        status = f"✅ Video processed! {len(embeddings_data['frame_paths'])} frames indexed"
        
        # Get first frame for preview
        preview = Image.open(embeddings_data['frame_paths'][0])
        
        return preview, status, embeddings_data['frame_paths']
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", []

def search_query(query, top_k=5):
    """Search for frames matching the query"""
    global searcher
    
    if searcher is None:
        return [], "⚠️ Please upload and process a video first"
    
    if not query or not query.strip():
        return [], "⚠️ Please enter a search query"
    
    results = searcher.search(query, top_k)
    
    if not results:
        return [], "🔍 No matching frames found. Try a different query."
    
    # Load result images
    result_images = []
    result_texts = []
    
    for i, result in enumerate(results):
        try:
            img = Image.open(result['frame_path'])
            result_images.append(img)
            result_texts.append(f"Match {i+1}: Score {result['similarity_score']:.3f}")
        except:
            continue
    
    status = f"✅ Found {len(results)} matches for: '{query}'"
    
    return result_images, status

# Create Gradio interface
with gr.Blocks(title="CLIP Video Retrieval", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 CLIP Video Retrieval System
    
    Search for specific moments in a video using **natural language descriptions**.
    
    ### How it works:
    1. **Upload a video** - The system extracts frames and generates CLIP embeddings
    2. **Enter a text query** - Describe what you want to find (e.g., "a person riding a bicycle")
    3. **Get matching frames** - Returns the most relevant frames from your video
    
    ### Example queries:
    - "a dog playing"
    - "sunset over the ocean" 
    - "a person wearing a red shirt"
    - "car driving fast"
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", height=300)
            frame_interval = gr.Slider(
                label="Frame Extraction Interval", 
                minimum=10, 
                maximum=100, 
                value=30, 
                step=10,
                info="Extract every N frames (lower = more frames, better search but slower)"
            )
            process_btn = gr.Button("📤 Process Video", variant="primary")
            
            status_text = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=1):
            preview_image = gr.Image(label="Video Preview", height=300)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="🔍 Search Query", 
                placeholder="Describe what you want to find (e.g., 'a person walking on the beach')",
                lines=2
            )
            top_k = gr.Slider(label="Number of Results", minimum=1, maximum=10, value=5, step=1)
            search_btn = gr.Button("🔎 Search", variant="primary")
        
        with gr.Column():
            result_images = gr.Gallery(label="Search Results", columns=5, rows=1, height=200)
            search_status = gr.Textbox(label="Search Status", interactive=False)
    
    # Examples
    gr.Markdown("### 💡 Example Queries to Try")
    gr.Examples(
        examples=[
            ["person smiling"],
            ["group of people"],
            ["outdoor scene"],
            ["close up face"],
            ["car or vehicle"],
        ],
        inputs=query_input
    )
    
    # Connect functions
    process_btn.click(
        fn=process_video,
        inputs=[video_input, frame_interval],
        outputs=[preview_image, status_text, gr.State()]
    )
    
    search_btn.click(
        fn=search_query,
        inputs=[query_input, top_k],
        outputs=[result_images, search_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
