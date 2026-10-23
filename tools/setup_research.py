"""Fetch the official research source and weights. No video uploads or credentials."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import time

ROOT=Path(__file__).resolve().parent.parent
repo=ROOT/"vendor"/"ProPainter"
RESEARCH_COMMIT="e870e79321c31b733e2031af5aa2fb1fe3ac7eec"


def download(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>1000:
        return
    for attempt in range(3):
        try:
            print(f"Downloading {dest.name}",flush=True)
            partial=dest.with_suffix(dest.suffix+".part")
            urllib.request.urlretrieve(url,partial)
            partial.replace(dest)
            return
        except Exception:
            if attempt==2:
                raise
            time.sleep(2)


if not (repo/"inference_propainter.py").exists():
    archive=ROOT/"vendor"/"propainter-source.zip"
    download(f"https://codeload.github.com/sczhou/ProPainter/zip/{RESEARCH_COMMIT}",archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(archive.parent/member.filename).resolve()
            if not target.is_relative_to(archive.parent.resolve()):
                raise RuntimeError("Unsafe archive member")
        z.extractall(archive.parent)
    (archive.parent/f"ProPainter-{RESEARCH_COMMIT}").rename(repo)
    (repo/"source_version.json").write_text(json.dumps({"commit":RESEARCH_COMMIT,
        "url":f"https://github.com/sczhou/ProPainter/commit/{RESEARCH_COMMIT}"},indent=2),encoding="utf-8")
records=[]
for name in ("raft-things.pth","recurrent_flow_completion.pth","ProPainter.pth"):
    dest=repo/"weights"/name
    download("https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"+name,dest)
    records.append({"name":name,"bytes":dest.stat().st_size,"sha256":hashlib.sha256(dest.read_bytes()).hexdigest()})
(repo/"download_manifest.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print("Research source and weights ready",flush=True)
