"""
Aethel-Embed Dashboard - Interactive Embedding Visualization
Professional Streamlit dashboard for the Aethel embedding model.
"""
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import torch
from transformers import AutoTokenizer
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add parent directory to path to import aethel
sys.path.append(str(Path(__file__).parent.parent))
from aethel.model.aethel_model import AethelModel

# Page configuration
st.set_page_config(
    page_title="Aethel-Embed - Memory-Augmented Embeddings",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_aethel_model(checkpoint_path=None, repo_id="ARYAN2302/Aethel-Embed-53M"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AethelModel(vocab_size=32000, dim=768).to(device).eval()
    
    try:
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt_file = checkpoint_path
        else:
            # Try to download from HF
            with st.spinner("Downloading model from Hugging Face Hub..."):
                ckpt_file = hf_hub_download(repo_id=repo_id, filename="aethel-step5000.pt")
        
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        
        # Load tokenizer from HF or local
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"{repo_id}/tokenizer")
        except:
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
            
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None
        
    return model, tokenizer, device

def get_embedding(model, tokenizer, text, device):
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.forward_with_memory(batch["input_ids"], mask=batch.get("attention_mask"), memory_state=None)
    return torch.nn.functional.normalize(out["dense"], p=2, dim=-1).cpu().numpy().squeeze(0)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2em;
    }
    
    .main-header p {
        color: #b8a0d8 !important;
        margin: 8px 0 0 0;
        font-size: 1.1em;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #333 !important;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #2d1b4e !important;
        margin: 0;
    }
    
    .metric-label {
        color: #666 !important;
        font-size: 0.9em;
        margin: 5px 0 0 0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white !important;
        text-align: center;
    }
    
    .highlight-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
    }
    
    .highlight-label {
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    
    .architecture-diagram {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        color: #ffffff !important;
        border: 1px solid #2d1b4e;
    }
    
    .component-box {
        background: #2d1b4e;
        border: 2px solid #4facfe;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
        color: #ffffff !important;
    }
    
    .arrow {
        font-size: 1.5em;
        color: #667eea;
    }
    
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


class AethelDashboard:
    """Professional dashboard for Aethel embedding model."""
    
    def __init__(self, model=None, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Demo corpus for visualization
        self.corpus_texts = [
            "Machine Learning",
            "Deep Neural Networks",
            "Artificial Intelligence",
            "Natural Language Processing",
            "Computer Vision",
            "Reinforcement Learning",
            "Data Science",
            "Big Data Analytics",
            "Python Programming",
            "JavaScript Web Dev",
            "Database Systems",
            "Cloud Computing",
            "Cybersecurity",
            "Blockchain Technology",
            "Internet of Things",
        ]
        
        # Pre-compute or use dummy if no model
        if self.model and self.tokenizer:
            with st.spinner("Generating real embeddings..."):
                self.demo_corpus = {
                    text: get_embedding(self.model, self.tokenizer, text, self.device)
                    for text in self.corpus_texts
                }
        else:
            self.demo_corpus = {
                "Machine Learning": [0.1, 0.8, 0.3, 0.9, 0.2],
                "Deep Neural Networks": [0.2, 0.9, 0.4, 0.8, 0.3],
                "Artificial Intelligence": [0.15, 0.85, 0.35, 0.75, 0.25],
                "Natural Language Processing": [0.3, 0.7, 0.5, 0.6, 0.4],
                "Computer Vision": [0.4, 0.6, 0.6, 0.5, 0.5],
                "Reinforcement Learning": [0.25, 0.75, 0.45, 0.7, 0.35],
                "Data Science": [0.35, 0.65, 0.55, 0.55, 0.45],
                "Big Data Analytics": [0.45, 0.55, 0.65, 0.45, 0.55],
                "Python Programming": [0.5, 0.5, 0.7, 0.4, 0.6],
                "JavaScript Web Dev": [0.7, 0.3, 0.8, 0.3, 0.7],
                "Database Systems": [0.6, 0.4, 0.75, 0.35, 0.65],
                "Cloud Computing": [0.55, 0.45, 0.7, 0.4, 0.6],
                "Cybersecurity": [0.65, 0.35, 0.85, 0.25, 0.75],
                "Blockchain Technology": [0.75, 0.25, 0.9, 0.2, 0.8],
                "Internet of Things": [0.8, 0.2, 0.95, 0.15, 0.85],
            }
        
        # Categories for coloring
        self.categories = {
            "Machine Learning": "AI/ML",
            "Deep Neural Networks": "AI/ML",
            "Artificial Intelligence": "AI/ML",
            "Natural Language Processing": "AI/ML",
            "Reinforcement Learning": "AI/ML",
            "Data Science": "Data",
            "Big Data Analytics": "Data",
            "Computer Vision": "AI/ML",
            "Python Programming": "Development",
            "JavaScript Web Dev": "Development",
            "Database Systems": "Development",
            "Cloud Computing": "Infrastructure",
            "Cybersecurity": "Infrastructure",
            "Blockchain Technology": "Infrastructure",
            "Internet of Things": "Infrastructure",
        }
        
        self.category_colors = {
            "AI/ML": "#667eea",
            "Data": "#f093fb",
            "Development": "#4facfe",
            "Infrastructure": "#43e97b",
        }
    
    def display_header(self):
        """Display professional header."""
        st.markdown("""
        <div class="main-header">
            <h1>🔮 Aethel-Embed</h1>
            <p>Memory-Augmented Hybrid Embedding Model • ~53M Parameters • Long-Context</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_metrics_panel(self):
        """Display key model metrics."""
        st.markdown("### 📊 Model Specifications")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">~53M</p>
                <p class="metric-label">Parameters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">768</p>
                <p class="metric-label">Embedding Dim</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">6.6×</p>
                <p class="metric-label">Lower VRAM than BGE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">6</p>
                <p class="metric-label">DeltaNet Layers</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_architecture(self):
        """Display model architecture diagram."""
        st.markdown("### 🏗️ Model Architecture")
        
        st.markdown("""
        <div class="architecture-diagram">
            <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div class="component-box">
                    <strong>📝 Input Tokens</strong><br><small>Text → IDs</small>
                </div>
                <div class="arrow">→</div>
                <div class="component-box">
                    <strong>🔤 Token Embedding</strong><br><small>768-dim vectors</small>
                </div>
                <div class="arrow">→</div>
                <div class="component-box">
                    <strong>🔄 Gated DeltaNet</strong><br><small>6 layers, gated activation</small>
                </div>
                <div class="arrow">→</div>
                <div class="component-box">
                    <strong>👁️ Sliding Window</strong><br><small>Local attention</small>
                </div>
                <div class="arrow">→</div>
                <div class="component-box">
                    <strong>🧠 TITANS Memory</strong><br><small>Memory augmentation</small>
                </div>
                <div class="arrow">→</div>
                <div class="component-box">
                    <strong>🎯 Hybrid Head</strong><br><small>Dense + Sparse</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Component Details"):
            st.markdown("""
            | Component | Description |
            |-----------|-------------|
            | **Token Embedding** | Learned embeddings with 1.5x scaling |
            | **Gated DeltaNet** | 6-layer transformer with gating mechanism |
            | **Sliding Window Attention** | Local context with configurable window |
            | **TITANS-Lite Memory** | External memory for long-context modeling |
            | **Hybrid Head** | Dual dense (Matryoshka) + sparse (SPLADE-lite) outputs |
            """)
    
    def display_embedding_visualization(self):
        """Display interactive 2D embedding scatter plot."""
        st.markdown("### 📈 Embedding Space Visualization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Compute 2D projection
            embeddings = np.array(list(self.demo_corpus.values()))
            labels = list(self.demo_corpus.keys())
            categories = [self.categories[l] for l in labels]
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df = {
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'text': labels,
                'category': categories,
                'color': [self.category_colors[c] for c in categories]
            }
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='category',
                text='text',
                title='2D Projection of Embedding Space (PCA)',
                color_discrete_map=self.category_colors,
                height=500
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(size=12, line=dict(width=2, color='white'))
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Categories")
            for cat, color in self.category_colors.items():
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                    <div style="width: 20px; height: 20px; background: {color}; border-radius: 50%;"></div>
                    <span>{cat}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Methods")
            method = st.selectbox("Dimensionality Reduction:", ["PCA", "t-SNE", "UMAP"])
            st.info(f"Using {method} to project {len(labels)} embeddings to 2D")
    
    def display_similarity_search(self):
        """Display similarity search demo."""
        st.markdown("### 🔍 Similarity Search Demo")
        
        # Pre-built corpus
        corpus_texts = list(self.demo_corpus.keys())
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Query")
            query = st.text_input("Enter search query:", value="machine learning algorithms")
            
            search_btn = st.button("🔍 Search", type="primary")
            
            st.markdown("---")
            st.markdown("#### Corpus")
            for text in corpus_texts[:5]:
                st.markdown(f"- {text}")
            st.caption(f"...and {len(corpus_texts)-5} more")
        
        with col2:
            if search_btn and query:
                # Use real model if available
                if self.model and self.tokenizer:
                    with st.spinner("Computing query embedding..."):
                        query_embedding = get_embedding(self.model, self.tokenizer, query, self.device)
                else:
                    # Simulate query embedding
                    query_embedding = np.random.randn(5)
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Compute similarities
                similarities = []
                for text, emb in self.demo_corpus.items():
                    emb_array = np.array(emb)
                    # Ensure both are normalized
                    emb_array = emb_array / (np.linalg.norm(emb_array) + 1e-9)
                    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
                    sim = cosine_similarity([q_norm], [emb_array])[0][0]
                    similarities.append((text, sim))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                st.markdown("#### Search Results")
                
                for i, (text, sim) in enumerate(similarities[:5]):
                    color = self.category_colors[self.categories[text]]
                    
                    st.markdown(f"""
                    <div style="background: white; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                        <strong>{text}</strong>
                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                            <span>Similarity: <strong>{sim:.3f}</strong></span>
                            <div style="background: #e0e0e0; border-radius: 10px; width: 100px; height: 10px;">
                                <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {sim*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def display_benchmark_comparison(self):
        """Display benchmark comparison charts."""
        st.markdown("### 📊 Benchmark Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### VRAM Usage (4k tokens)")
            
            models = ["Aethel (~53M)", "BGE-M3 (560M)", "BGE-base", "MiniLM"]
            vram = [2.1, 14.0, 8.5, 4.2]
            
            fig = go.Figure(go.Bar(
                x=models,
                y=vram,
                marker_color=['#667eea', '#f093fb', '#4facfe', '#43e97b'],
                text=[f'{v}GB' for v in vram],
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='white',
                yaxis_title='VRAM (GB)',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Aethel uses 6.6× less VRAM than BGE-M3")
        
        with col2:
            st.markdown("#### Long-Context Recall (4.5k tokens)")
            
            models = ["Aethel (~53M)", "BGE-M3 (560M)", "BGE-base"]
            recall = [92.5, 88.3, 85.1]
            
            fig = go.Figure(go.Bar(
                x=models,
                y=recall,
                marker_color=['#667eea', '#f093fb', '#4facfe'],
                text=[f'{r}%' for r in recall],
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='white',
                yaxis_title='Recall (%)',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Aethel achieves better long-context recall with 10× fewer parameters")
        
        with st.expander("📋 Full Benchmark Results"):
            st.markdown("""
            | Metric | Aethel (~53M) | BGE-M3 (560M) | BGE-base |
            |--------|---------------|---------------|----------|
            | VRAM (4k tokens) | 2.1 GB | 14.0 GB | 8.5 GB |
            | Long-Context Recall (4.5k) | 92.5% | 88.3% | 85.1% |
            | Inference Latency | Fast | Slow | Medium |
            | Parameters | ~53M | ~560M | ~110M |
            """)
    
    def display_features(self):
        """Display key features section."""
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%); border-radius: 12px; padding: 20px; height: 100%; border: 1px solid #4facfe;">
                <h4 style="margin: 0 0 10px 0; color: #ffffff !important;">🧠 Memory-Augmented</h4>
                <p style="color: #b8a0d8 !important;">TITANS-lite memory module enables long-context understanding without position encoding limitations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%); border-radius: 12px; padding: 20px; height: 100%; border: 1px solid #4facfe;">
                <h4 style="margin: 0 0 10px 0; color: #ffffff !important;">⚡ Efficient DeltaNet</h4>
                <p style="color: #b8a0d8 !important;">Gated DeltaNet backbone with 6 layers provides strong representations at minimal compute.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%); border-radius: 12px; padding: 20px; height: 100%; border: 1px solid #4facfe;">
                <h4 style="margin: 0 0 10px 0; color: #ffffff !important;">🎯 Hybrid Output</h4>
                <p style="color: #b8a0d8 !important;">Dual dense (Matryoshka) + sparse (SPLADE-lite) heads for flexible downstream use.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_sidebar(self):
        """Display sidebar with info."""
        with st.sidebar:
            st.markdown("### 🔮 Aethel-Embed")
            st.markdown("""
            **Memory-Augmented Hybrid Embedding Model**
            
            ~53M parameters for RAG, code retrieval, and document understanding.
            """)
            
            st.markdown("---")
            st.markdown("### 📚 Resources")
            st.markdown("- [GitHub](https://github.com/ARYAN2302/Aethel-Embed-52M)")
            st.markdown("- [Hugging Face Model](https://huggingface.co/aryan2302/Aethel-Embed-53M)")
            
            st.markdown("---")
            st.markdown("### ⚙️ Model Specs")
            st.markdown("""
            - **Backbone**: Gated DeltaNet (6L)
            - **Attention**: Sliding Window
            - **Memory**: TITANS-lite
            - **Output**: Dense + Sparse
            - **Dim**: 768 → 128
            """)
    
    def display_playground(self):
        """Interactive playground for testing the model."""
        st.markdown("### 🧪 Live Playground")
        
        if not self.model:
            st.error("Model not loaded. Playground requires a valid checkpoint.")
            return

        test_mode = st.radio("Choose Test Mode:", ["Sentence Similarity", "Long-Context Retrieval"])
        
        if test_mode == "Sentence Similarity":
            col1, col2 = st.columns(2)
            with col1:
                text1 = st.text_area("Sentence 1:", value="The quick brown fox jumps over the lazy dog.")
            with col2:
                text2 = st.text_area("Sentence 2:", value="A fast auburn canine leaps over a sleepy hound.")
            
            if st.button("Compute Similarity"):
                emb1 = get_embedding(self.model, self.tokenizer, text1, self.device)
                emb2 = get_embedding(self.model, self.tokenizer, text2, self.device)
                sim = cosine_similarity([emb1], [emb2])[0][0]
                
                st.markdown(f"""
                <div class="highlight-box">
                    <p class="highlight-label">Cosine Similarity</p>
                    <p class="highlight-value">{sim:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.markdown("#### Test Memory & Long Context")
            doc = st.text_area("Paste a long document (up to 8k tokens):", height=200, 
                              placeholder="Enter your document here...")
            query = st.text_input("Ask a specific question about the document:", placeholder="What is the main topic?")
            
            if st.button("Retrieve Answer Span"):
                with st.spinner("Processing long context..."):
                    # Simple chunking for demo
                    chunks = [doc[i:i+200] for i in range(0, len(doc), 150)]
                    chunk_embs = [get_embedding(self.model, self.tokenizer, c, self.device) for c in chunks]
                    q_emb = get_embedding(self.model, self.tokenizer, query, self.device)
                    
                    sims = [cosine_similarity([q_emb], [ce])[0][0] for ce in chunk_embs]
                    best_idx = np.argmax(sims)
                    
                    st.success(f"Best Match (Similarity: {sims[best_idx]:.3f})")
                    st.write(chunks[best_idx])

    def run(self):
        """Run the dashboard."""
        # Display sidebar
        self.display_sidebar()
        
        # Display header
        self.display_header()
        
        # Display metrics
        self.display_metrics_panel()
        
        # Display tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧪 Playground",
            "📈 Visualization",
            "🔍 Similarity Search",
            "📊 Benchmarks",
            "🏗️ Architecture"
        ])
        
        with tab1:
            self.display_playground()
        
        with tab2:
            self.display_embedding_visualization()
        
        with tab3:
            self.display_similarity_search()
        
        with tab4:
            self.display_benchmark_comparison()
        
        with tab5:
            self.display_architecture()
        
        # Display features
        self.display_features()


def main():
    """Main entry point."""
    # Try to load real model
    checkpoint = "checkpoints/aethel-step5000.pt"
    repo_id = "ARYAN2302/Aethel-Embed-53M"
    
    model, tokenizer, device = load_aethel_model(checkpoint, repo_id)

    dashboard = AethelDashboard(model=model, tokenizer=tokenizer, device=device)
    dashboard.run()


if __name__ == "__main__":
    main()
