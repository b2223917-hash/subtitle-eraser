"""Isolated offline ProPainter benchmark using the exact OpenCV segmentation masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

from .pipeline import repair
from .validate import write_report, make_browser_video, summary

ROOT = Path(__file__).resolve().parent.parent


def read_image(path,flags=cv2.IMREAD_COLOR):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),flags)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def run_research(output: Path, research_python: Path, repo: Path, clips=None):
    output, repo = output.resolve(), repo.resolve()
    wrapper = ROOT/"tools"/"propainter_entry.py"
    metrics = {"mode":"offline, includes model initialization and encoding; downloads excluded",
               "clips":[]}
    gpu = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True)
    metrics["gpu"] = gpu.stdout.strip()
    for clip in clips or ("simple","face","motion"):
        source = output/"research_inputs"/clip
        inputs = sorted((source/"frames").glob("*.png"))
        if not inputs:
            raise RuntimeError("请先生成完整的十二秒验证素材")
        destination = output/"propainter"/clip
        destination.mkdir(parents=True,exist_ok=True)
        command = [str(research_python.resolve()),str(wrapper),"--repo",str(repo),
                   "--video",str(source/"frames"),"--mask",str(source/"masks"),
                   "--output",str(destination),"--save_fps","24","--fp16","--save_frames",
                   "--mask_dilation","0","--neighbor_length","6","--ref_stride","6",
                   "--subvideo_length","24","--raft_iter","12"]
        started = perf_counter()
        log_path = destination/"inference.log"
        with log_path.open("w",encoding="utf-8") as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
            memory = []
            while process.poll() is None:
                stats = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                                       capture_output=True,text=True,creationflags=0x08000000)
                try:
                    memory.append(float(stats.stdout.strip().splitlines()[0]))
                except (ValueError,IndexError):
                    pass
                try:
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
        elapsed = perf_counter()-started
        item = {"clip":clip,"frames":len(inputs),"resolution":[576,160],"elapsed_s":elapsed,
                "offline_fps_including_startup":len(inputs)/elapsed,
                "sampled_total_gpu_used_peak_mib":max(memory) if memory else None,
                "return_code":process.returncode,"log":str(log_path.relative_to(output))}
        native_stats = destination/"native_stats.json"
        if native_stats.exists():
            item.update(json.loads(native_stats.read_text(encoding="utf-8")))
        if process.returncode == 0:
            predictions = sorted((destination/"frames"/"frames").glob("*.png"))
            if len(predictions) != len(inputs):
                raise RuntimeError(f"ProPainter输出帧数不匹配：{len(predictions)} / {len(inputs)}")
            comparison = destination/"comparison.mp4"
            writer = cv2.VideoWriter(str(comparison),cv2.VideoWriter_fourcc(*"mp4v"),24,(1728,192))
            light_times, light_errors, model_errors = [],[],[]
            untouched_changes = 0
            for original_file,prediction_file in zip(inputs,predictions):
                original = read_image(original_file)
                mask = read_image(source/"masks"/original_file.name,0)
                clean = read_image(source/"clean"/original_file.name)
                inferred = read_image(prediction_file)
                # Explicitly composite through the original mask after model resizing.
                inferred[mask==0] = original[mask==0]
                tick = perf_counter()
                light = repair(original,mask)
                light_times.append((perf_counter()-tick)*1000)
                changed = np.any(inferred!=original,axis=2)
                untouched_changes += int(np.count_nonzero(changed&(mask==0)))
                if np.any(mask):
                    light_errors.append(float(np.abs(light[mask>0].astype(float)-clean[mask>0]).mean()))
                    model_errors.append(float(np.abs(inferred[mask>0].astype(float)-clean[mask>0]).mean()))
                panels=[]
                for image,text in ((original,"ORIGINAL"),(light,"OPENCV"),(inferred,"PROPAINTER / OFFLINE")):
                    panel=cv2.copyMakeBorder(image,32,0,0,0,cv2.BORDER_CONSTANT,value=(26,34,48))
                    cv2.putText(panel,text,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
                    panels.append(panel)
                writer.write(np.concatenate(panels,axis=1))
            writer.release()
            make_browser_video(comparison)
            item.update({"comparison":str(comparison.relative_to(output)).replace("\\","/"),
                         "opencv_repair_only_ms":summary(light_times),
                         "opencv_mae_inside_mask":float(np.mean(light_errors)),
                         "propainter_mae_inside_mask":float(np.mean(model_errors)),
                         "outside_mask_changed_pixels":untouched_changes})
        else:
            item["error"] = log_path.read_text(encoding="utf-8",errors="replace")[-2500:]
        metrics["clips"].append(item)
        (output/"propainter_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
        write_report(output)
        print(json.dumps(item,ensure_ascii=True,indent=2),flush=True)
    return metrics


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"validation")
    parser.add_argument("--python",type=Path,default=ROOT/".research-venv"/"Scripts"/"python.exe")
    parser.add_argument("--repo",type=Path,default=ROOT/"vendor"/"ProPainter")
    parser.add_argument("--clips",nargs="*")
    args=parser.parse_args()
    run_research(args.output,args.python,args.repo,args.clips)
