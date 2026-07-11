"""Step 1: Download PLAsTiCC dataset"""
import sys, os
sys.path.insert(0, r"D:\AITools\AstroTransient")
os.chdir(r"D:\AITools\AstroTransient")

from src.data.download import download_plasticc
download_plasticc("data/raw/plasticc")
print("Download complete!")
