from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path
from urllib.request import urlopen, Request

URL = "https://github.com/myrosia/semeval-2013-task7/raw/refs/heads/main/semeval-3way.zip"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='data/raw')
    ap.add_argument('--url', default=URL)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    req=Request(args.url, headers={'User-Agent':'ED05-data-downloader/1.0'})
    with urlopen(req, timeout=60) as r:
        data=r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    print(f"Extracted {len(z.namelist())} files to {out}")

if __name__=='__main__': main()
