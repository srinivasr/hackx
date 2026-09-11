#!/usr/bin/env python3
"""Fetch authentic clinical Dermatology (HAM10000) and Histopathology (PCam/IDC) images from Kaggle
and populate data/sample_dermatology_scans and data/sample_histopathology_scans.
Uses HTTP Range requests to selectively extract only the target 60 images for each modality.
"""

import csv
import io
import os
import sys
import urllib.request
import zipfile
import cv2
import numpy as np
import pandas as pd


class RemoteZipFile(io.RawIOBase):
    """Seekable stream that reads chunks from a remote HTTP file via Range requests."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers["content-length"])
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            end = self.length - 1
        else:
            end = min(self.pos + size - 1, self.length - 1)
        if self.pos >= self.length:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def populate_dermatology(archive_url: str):
    print("\n--- Ingesting Dermatology (HAM10000) from Kaggle ---")
    rz = RemoteZipFile(archive_url)
    zf = zipfile.ZipFile(rz)
    all_names = zf.namelist()

    # Read HAM10000_metadata.csv
    meta_bytes = zf.read("HAM10000_metadata.csv")
    meta_text = meta_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(meta_text))

    name_set = set(all_names)

    mel_files = []
    benign_files = []

    for row in reader:
        img_id = row["image_id"]
        dx = row["dx"]
        p1 = f"HAM10000_images_part_1/{img_id}.jpg"
        p2 = f"HAM10000_images_part_2/{img_id}.jpg"
        path = p1 if p1 in name_set else (p2 if p2 in name_set else None)
        if not path:
            continue

        if dx == "mel" and len(mel_files) < 30:
            mel_files.append(path)
        elif dx in ["nv", "bkl"] and len(benign_files) < 45:
            benign_files.append(path)

        if len(mel_files) >= 24 and len(benign_files) >= 36:
            break

    print(f"Discovered {len(mel_files)} melanoma candidates and {len(benign_files)} benign candidates.")

    meta_df = pd.read_csv("data/sample_dermatology_metadata.csv")
    out_dir = "data/sample_dermatology_scans"
    os.makedirs(out_dir, exist_ok=True)

    mel_idx = 0
    benign_idx = 0

    for i, row in meta_df.iterrows():
        gt = int(row["ground_truth"])
        fn = f"lesion_{i+1:04d}.png"
        out_p = os.path.join(out_dir, fn)

        if gt == 1:
            src = mel_files[mel_idx]
            mel_idx += 1
            label_str = "MELANOMA (1)"
        else:
            src = benign_files[benign_idx]
            benign_idx += 1
            label_str = "BENIGN (0)"

        raw = zf.read(src)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_p, img_resized)
        print(f"  [Derm {i+1:02d}/60] {fn} <- {os.path.basename(src)} [{label_str}]")

    print("✅ Successfully populated 60 authentic HAM10000 dermoscopy lesions.")


def populate_histopathology(archive_url: str):
    print("\n--- Ingesting Histopathology (BreakHis 700x460 H&E Slides) from Kaggle ---")
    rz = RemoteZipFile(archive_url)
    zf = zipfile.ZipFile(rz)
    all_names = zf.namelist()

    benign_candidates = sorted([
        n for n in all_names
        if "/benign/" in n and "/100X/" in n and n.endswith(".png")
    ])[:50]
    malignant_candidates = sorted([
        n for n in all_names
        if "/malignant/" in n and "/100X/" in n and n.endswith(".png")
    ])[:50]

    print(f"Discovered {len(benign_candidates)} benign and {len(malignant_candidates)} malignant candidates from BreakHis.")

    meta_df = pd.read_csv("data/sample_histopathology_metadata.csv")
    out_dir = "data/sample_histopathology_scans"
    os.makedirs(out_dir, exist_ok=True)

    b_idx = 0
    m_idx = 0

    for i, row in meta_df.iterrows():
        gt = int(row["ground_truth"])
        fn = f"wsi_{i+1:04d}.png"
        out_p = os.path.join(out_dir, fn)

        if gt == 1:
            src = malignant_candidates[m_idx]
            m_idx += 1
            label_str = "CARCINOMA (1)"
        else:
            src = benign_candidates[b_idx]
            b_idx += 1
            label_str = "BENIGN (0)"

        raw = zf.read(src)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_p, img_resized)
        print(f"  [Path {i+1:02d}/60] {fn} <- {os.path.basename(src)} [{label_str}]")

    print("✅ Successfully populated 60 authentic BreakHis histopathology slides.")


if __name__ == "__main__":
    ham_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://storage.googleapis.com/kaggle-data-sets/54339/104884/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260911T173753Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0f02289490f979ff18dfb8696e6198f1f59215eeadeca814e64a048078dc94d5ddd22e0cd9f94315869e7cad52be80df1eb8beda048362ab2fc44d9972c0a0ecf92fae202b17983605f2dc31d640cb301759ee23b5a472af25a3fca08d45ee0b6a994db1c28c9f2fbb560b8832234e7267361bd61eb54f57c563219c64a04b7798413ced80de4fb864184cfd8209fc23d62a2fc68fb9ab26a3ad9b9181b87b3063d01bd3865904c741e914a53bd7b5ec70aedd9b34083c626730141d818c81ac0b53d1a00b6bb5eb92f630d1e3507a90c58978fabae4f070dcfe71a7d56dfce3e0a6ed415636fb7fed0da8c40645c082a881fca1b461fb300ebe399c998c7b54"
    )
    hist_url = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "https://storage.googleapis.com/kaggle-data-sets/209316/999617/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260911T174559Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=81b7e6da8de1713c57c180901c4205c8123f2b8a8129eac8cc48f9e279c4138d2ca298485e59e5fe5ad44803aa27d2914ada8ff6937e31b4b0b97d9117344a312c725c5e9fef06c0feed1731209d84691b2617452ccb9029b6eedb66235dd58d4bcb61ca053819eeed2ba7ae620a42850fcff7efb1184f43b5acc238ce17d2dc7b5fab4e6ad7ae04e8af930f013237b2d159c570fa8f24037a124aaf248f79830e76001b2806387941ada232e6ac6ae57e368f0a3646a209cc1ae773020dfa18a91487dd8ad9f4b086f5597f3588e23e48a8792343c6c530b91e98161def717007fad61b87cd25246fc79e46e85931fcd3d074d513ad16ba734b5b680ced54f9"
    )
    # Only populate histopathology if run without args, since dermatology was already populated
    populate_histopathology(hist_url)
