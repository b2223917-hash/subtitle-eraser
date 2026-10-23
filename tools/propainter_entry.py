"""Bootstrap vendor code and record allocator peak without modifying the vendor."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--repo",required=True,type=Path)
args,remaining=parser.parse_known_args()
repo=args.repo.resolve()
sys.path.insert(0,str(repo))
os.chdir(repo)
import torch
torch.set_num_threads(2)
if not torch.cuda.is_available():
    raise RuntimeError("GPU研究环境未检测到CUDA，停止运行以避免误报CPU结果")
torch.cuda.reset_peak_memory_stats()
output=Path(remaining[remaining.index("--output")+1])
# OpenCV 4.11 on Windows cannot open Unicode paths through its narrow C API.
# Keep vendor code untouched and adapt only its image I/O boundary.
import cv2
import numpy as np


def unicode_read(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)


def unicode_write(path, image, params=None):
    ok,encoded=cv2.imencode(Path(path).suffix,image,params or [])
    if ok:
        encoded.tofile(path)
    return bool(ok)


cv2.imread=unicode_read
cv2.imwrite=unicode_write
sys.argv=[str(repo/"inference_propainter.py"),*remaining]
torch.cuda.synchronize()
started=perf_counter()
runpy.run_path(str(repo/"inference_propainter.py"),run_name="__main__")
torch.cuda.synchronize()
(output/"native_stats.json").write_text(json.dumps({
    "torch_gpu":torch.cuda.get_device_name(),
    "torch_version":torch.__version__,
    "inference_script_elapsed_s":perf_counter()-started,
    "torch_peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
    "torch_peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2},indent=2),encoding="utf-8")
