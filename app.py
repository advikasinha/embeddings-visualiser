# import os
# import shutil
# import re
# import numpy as np
# import pandas as pd
# import torch
# import streamlit as st
# import umap
# import plotly.express as px
# from datasets import load_dataset
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from models import InferSent

# st.title("InferSent Text Embeddings Visualization")

# # Install required packages
# os.system("pip install --upgrade packaging")
# os.system("pip install datasets --upgrade")

# # Copy necessary files and folders
# shutil.copytree("/kaggle/input/infersent/", "/kaggle/working/infersent")
# os.system("mv /kaggle/working/infersent/* /kaggle/working/")

# # Download InferSent model
# os.system("mkdir encoder")
# os.system("curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl")

# # Load Model
# model_version = 1
# MODEL_PATH = "encoder/infersent%s.pkl" % model_version
# W2V_PATH = '/kaggle/input/glove-840b-300d/glove.840B.300d.txt'
# VOCAB_SIZE = 1e5  # Load embeddings of VOCAB_SIZE most frequent words
# USE_CUDA = torch.cuda.is_available()

# params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
#                 'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
# model = InferSent(params_model)
# model.load_state_dict(torch.load(MODEL_PATH))
# model = model.cuda() if USE_CUDA else model
# model.set_w2v_path(W2V_PATH)
# model.build_vocab_k_words(K=VOCAB_SIZE)

# # Get Data
# dataset = load_dataset("recastai/coyo-75k-augmented-captions")
# sentences = dataset['train']['short_caption']
# sentences = [sentence[0] for sentence in sentences]
# sentences_ = [str(sentence) for sentence in sentences]

# # Apply the model embeddings
# embeddings = model.encode(sentences_, bsize=128, tokenize=False, verbose=True)
# df = pd.DataFrame(embeddings)
# df['prompt'] = sentences
# df.to_csv("infersent_embeddings.csv")

# # Load Embeddings
# embeddings = pd.read_csv("infersent_embeddings.csv")

# # Choose dimensionality reduction technique
# dim_reduction = st.selectbox("Choose Dimension Reduction Technique", ['PCA', 'UMAP', 'T-SNE'])

# # Dimensionality reduction functions
# def apply_pca(embeddings, n_components=3):
#     pca = PCA(n_components=n_components)
#     return pca.fit_transform(embeddings)

# def apply_umap(embeddings, n_components=3):
#     umap_reducer = umap.UMAP(n_components=n_components)
#     return umap_reducer.fit_transform(embeddings)

# def apply_tsne(embeddings, n_components=3):
#     tsne = TSNE(n_components=n_components)
#     return tsne.fit_transform(embeddings)

# # Apply chosen dimensionality reduction technique
# if dim_reduction == "PCA":
#     reduced_embeddings = apply_pca(embeddings.iloc[:, :-1])
# elif dim_reduction == "UMAP":
#     reduced_embeddings = apply_umap(embeddings.iloc[:, :-1])
# elif dim_reduction == "T-SNE":
#     reduced_embeddings = apply_tsne(embeddings.iloc[:, :-1])

# # Visualize reduced embeddings
# fig = px.scatter_3d(
#     x=reduced_embeddings[:, 0],
#     y=reduced_embeddings[:, 1],
#     z=reduced_embeddings[:, 2],
#     title=f"3D Scatter Plot of {dim_reduction} Reduced Data",
#     labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
#     opacity=0.7
# )
# st.plotly_chart(fig)

import os
import streamlit as st
import subprocess

# Ensure the setup is complete
def run_setup():
    if not os.path.exists("encoder/infersent1.pkl") or not os.path.exists("glove.840B.300d.txt"):
        subprocess.run(["python", "setup.py"], check=True)

# Run the setup
run_setup()

# Rest of the imports and code
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from models import InferSent

# Load dataset
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
