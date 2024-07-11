import os
import requests
import streamlit as st

def download_file(url, dest):
    if not os.path.exists(dest):
        st.write(f"Downloading {os.path.basename(dest)}...")
        response = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write(f"{os.path.basename(dest)} downloaded.")

def main():
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
        os.system("unzip glove.840B.300d.zip")
        st.write("GloVe embeddings unzipped.")
    
    st.write("Setup complete.")

if __name__ == "__main__":
    main()
