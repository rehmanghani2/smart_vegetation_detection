import os 
import gdown
import zipfile

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def downlaod_dataset(url, output, extract_to):
     """Download and extract dataset from Google Drive or URL"""
     if not os.path.exists(output):
         print(f"Downloading {output} ...")
         gdown.download(url, output, quiet=False)
     else:
         print(f"{output} already exits")
     # Extract
     
     with zipfile.ZipFile(output, "r") as zip_ref:
         zip_ref.extractall(extract_to)
     print(f"Extracted to {extract_to}")
     
# Example: EuroSAT (Google Drive mirror)
euroset_url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"         #  "https://drive.google.com/uc?id=1zJ3avRPO3pMb0YfDyDdMxRerjZ0BiA7d"
downlaod_dataset(euroset_url, os.path.join(DATA_DIR, "eurosat.zip"), os.path.join(DATA_DIR, "eurosat"))