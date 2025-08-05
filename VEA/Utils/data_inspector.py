from PIL import Image
from pathlib import Path
from collections import Counter
import os

class DataInspector:
    def __init__(self, data_folder: str, delete_corrupted: bool = True):
        self.data_folder = data_folder
        self.allowed_exts = [".jpg", ".jpeg", ".png"]
        self.sizes = []
        self.modes = []
        self.channels = []
        self.corrupted_files = []
        self.delete_corrupted = delete_corrupted

    def inspect(self):
        for file in Path(self.data_folder).rglob("*"):
            if file.suffix.lower() in self.allowed_exts:
                try:
                    img = Image.open(file)
                    img.verify()  # Quick corruption check
                    img = Image.open(file).convert("RGB")
                    self.sizes.append(img.size)
                    self.modes.append(img.mode)
                    self.channels.append(len(img.getbands()))
                except Exception as e:
                    self.corrupted_files.append(file)
                    if self.delete_corrupted:
                        try:
                            os.remove(file)
                            print(f"❌ Corrupted file removed: {file}")
                        except Exception as del_error:
                            print(f"⚠️ Couldn't delete {file}: {del_error}")

    def summary(self):
        avg_width = sum([w for w, h in self.sizes]) // len(self.sizes) if self.sizes else 0
        avg_height = sum([h for w, h in self.sizes]) // len(self.sizes) if self.sizes else 0

        print("\n✅ Dataset Summary")
        print(f"📁 Total valid images: {len(self.sizes)}")
        print(f"🖼️ Average image size: {avg_width} x {avg_height}")
        print(f"🎨 Channel distribution: {Counter(self.channels)}")
        print(f"🧾 Mode distribution: {Counter(self.modes)}")
        print(f"🚮 Corrupted files found: {len(self.corrupted_files)}")



