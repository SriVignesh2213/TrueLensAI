from torchvision import datasets
from pathlib import Path

train_dir = Path("c:/Users/SRI VIGNESH/Downloads/TrueLensAI/data/train").absolute()
print(f"Checking directory: {train_dir}")
try:
    ds = datasets.ImageFolder(str(train_dir))
    print(f"Success! Found {len(ds)} images in {len(ds.classes)} classes: {ds.classes}")
except Exception as e:
    print(f"Error loading ImageFolder: {e}")
