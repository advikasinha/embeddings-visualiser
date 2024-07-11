import os
import requests
import streamlit as st
import zipfile
import shutil
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from models import InferSent

def download_file(url, dest):
    if not os.path.exists(dest):
        st.write(f"Downloading {os.path.basename(dest)}...")
        response = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write(f"{os.path.basename(dest)} downloaded.")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    st.write(f"Extracted {zip_path} to {extract_to}")

# Ensure the setup is complete
def run_setup():
    if not os.path.exists("encoder/infersent1.pkl") or not os.path.exists("glove.840B.300d.txt"):
        st.write("Setting up environment...")
        
        # Create directories if they don't exist
        if not os.path.exists("encoder"):
            os.makedirs("encoder")
        
        # URLs for the files
        infersent_url = "https://dl.fbaipublicfiles.com/infersent/infersent1.pkl"
        glove_url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
        
        # Destinations
        infersent_dest = "encoder/infersent1.pkl"
        glove_dest = "glove.840B.300d.zip"
        
        # Download files
        download_file(infersent_url, infersent_dest)
        download_file(glove_url, glove_dest)
        
        # Unzip GloVe embeddings
        if not os.path.exists("glove.840B.300d.txt"):
            st.write("Unzipping GloVe embeddings...")
            unzip_file(glove_dest, ".")
            st.write("GloVe embeddings unzipped.")
        
        st.write("Setup complete.")

# Run the setup
run_setup()

# Rest of the imports and code
def load_data():
    dataset = load_dataset("recastai/coyo-75k-augmented-captions")
    sentences = dataset['train']['short_caption']
    sentences = [sentence[0] for sentence in sentences]
    return sentences

# Load model
def load_model():
    model_version = 1
    MODEL_PATH = "encoder/infersent1.pkl"
    W2V_PATH = 'glove.840B.300d.txt'
    VOCAB_SIZE = 1e5  # Load embeddings of VOCAB_SIZE most frequent words
    USE_CUDA = True  # Keep it on CPU if False, otherwise will put it on GPU

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda() if USE_CUDA else model
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=VOCAB_SIZE)
    return model

# Generate embeddings
def generate_embeddings(sentences, model):
    sentences_ = [str(sentence) for sentence in sentences]
    embeddings = model.encode(sentences_, bsize=128, tokenize=False, verbose=True)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['prompt'] = sentences
    return embeddings_df

# Visualize embeddings
def visualize_embeddings(embeddings):
    st.write("PCA Visualization")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings.iloc[:, :-1])
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["prompt"] = embeddings["prompt"]
    fig = px.scatter(pca_df, x="PCA1", y="PCA2", hover_data=["prompt"])
    st.plotly_chart(fig)

    st.write("t-SNE Visualization")
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings.iloc[:, :-1])
    tsne_df = pd.DataFrame(tsne_result, columns=["t-SNE1", "t-SNE2"])
    tsne_df["prompt"] = embeddings["prompt"]
    fig = px.scatter(tsne_df, x="t-SNE1", y="t-SNE2", hover_data=["prompt"])
    st.plotly_chart(fig)

    st.write("UMAP Visualization")
    umap_model = umap.UMAP(n_components=2)
    umap_result = umap_model.fit_transform(embeddings.iloc[:, :-1])
    umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
    umap_df["prompt"] = embeddings["prompt"]
    fig = px.scatter(umap_df, x="UMAP1", y="UMAP2", hover_data=["prompt"])
    st.plotly_chart(fig)

def main():
    st.title("Embedding Visualizer")

    st.write("Loading dataset...")
    sentences = load_data()

    st.write("Loading model...")
    model = load_model()

    st.write("Generating embeddings...")
    embeddings = generate_embeddings(sentences, model)

    st.write("Visualizing embeddings...")
    visualize_embeddings(embeddings)

if __name__ == "__main__":
    main()
