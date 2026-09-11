#!/usr/bin/env python3
"""Fetch 60 authentic clinical chest radiographs from Kaggle and populate data/sample_scans.
Uses HTTP Range requests to extract only the target 60 image files without downloading the 2.4 GB archive.
"""

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


def fetch_and_populate_cohort(
    archive_url: str,
    output_dir: str = "data/sample_scans",
    metadata_csv: str = "data/sample_metadata.csv",
):
    print(f"Connecting to Kaggle remote archive ({archive_url[:60]}...)...")
    remote_stream = RemoteZipFile(archive_url)
    print(f"Connected. Remote archive size: {remote_stream.length / (1024 * 1024):.1f} MB")

    print("Parsing remote central directory index...")
    zf = zipfile.ZipFile(remote_stream)
    all_names = zf.namelist()

    # Find normal and pneumonia files in test set (excluding macOS metadata)
    normal_files = sorted(
        [
            n
            for n in all_names
            if "test/NORMAL/" in n
            and "__MACOSX" not in n
            and "/._" not in n
            and n.endswith((".jpeg", ".jpg", ".png"))
        ]
    )
    pneumonia_files = sorted(
        [
            n
            for n in all_names
            if "test/PNEUMONIA/" in n
            and "__MACOSX" not in n
            and "/._" not in n
            and n.endswith((".jpeg", ".jpg", ".png"))
        ]
    )

    print(f"Discovered {len(normal_files)} normal candidates and {len(pneumonia_files)} pneumonia candidates.")

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file {metadata_csv} not found.")

    meta_df = pd.read_csv(metadata_csv)
    total_samples = len(meta_df)
    print(f"Target cohort size: {total_samples} from {metadata_csv}")

    os.makedirs(output_dir, exist_ok=True)

    normal_idx = 0
    pneumonia_idx = 0

    for i, row in meta_df.iterrows():
        gt = int(row["ground_truth"])
        scan_filename = f"scan_{i+1:04d}.png"
        out_path = os.path.join(output_dir, scan_filename)

        if gt == 1:
            source_zip_path = pneumonia_files[pneumonia_idx]
            pneumonia_idx += 1
            label_str = "PNEUMONIA (1)"
        else:
            source_zip_path = normal_files[normal_idx]
            normal_idx += 1
            label_str = "NORMAL (0)"

        raw_data = zf.read(source_zip_path)
        arr = np.frombuffer(raw_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to decode image from {source_zip_path}")

        # Standardize to 224x224
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Save to target file
        cv2.imwrite(out_path, img_resized)
        print(f"  [{i+1:02d}/{total_samples:02d}] {scan_filename} <- {os.path.basename(source_zip_path)} [{label_str}]")

    print(f"\n✅ Successfully populated {total_samples} authentic clinical radiographs in {output_dir}")
    print(f"   - Normal scans: {normal_idx}")
    print(f"   - Pneumonia scans: {pneumonia_idx}")


if __name__ == "__main__":
    archive_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://storage.googleapis.com/kaggle-data-sets/17810/23812/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260911T172441Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=99e79a42d91eff8e69d709535ffa6e03d8531b896834effa61bfc63c91c5e1e5f59c26db349b826385c6c993ec85b1d6863295bb82b66fa9f5c1880e98aa43e68881dda8d5b8eb86ab577f18599debaf0d8e6308b7138c4899fa2e2d00336421e26f599ad20eeb3ca5d9cdfde72bfd5ebc27a4b4b42ac87f55fc699e8c8cca0c93e4d6b46d2d2927e0827017033283d8124426a02439b503a18d7dccbe6506f341cf3f7f0f4adb1a48f87573c6bbb11cbb54dabdfa4d3177725c25c2e9ab8e5eebd5916ca79e794c781ebf8cff86886515f42d5d97b385865f11c96cedf918718fa1ed4b5480a3dbbf5c8abbc36e8774dbdbc666d7bb7f9d661d55197b476897"
    )
    fetch_and_populate_cohort(archive_url)
