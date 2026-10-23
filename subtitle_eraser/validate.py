"""Generate reproducible synthetic video comparisons and measured QA artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import html
import json
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from .fixtures import sample_frame, SCENES
from .pipeline import SubtitleProcessor, polygon_mask


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(path)


def summary(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "p50": float(np.percentile(values,50)),
            "p95": float(np.percentile(values,95)), "max": float(values.max())}


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def make_browser_video(path):
    import imageio_ffmpeg
    temp = path.with_name(path.stem + ".h264.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-i",str(path),
                    "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p", "-movflags", "+faststart", str(temp)],check=True)
    temp.replace(path)


def label(image, text):
    output = cv2.resize(image, (640, 360))
    output = cv2.copyMakeBorder(output, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(26, 34, 48))
    cv2.putText(output, text, (16,24), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,240,250), 1, cv2.LINE_AA)
    return output


def run_validation(dest: Path, seconds: float = 12):
    dest.mkdir(parents=True, exist_ok=True)
    fps = 24
    frames = int(seconds*fps)
    processor = SubtitleProcessor()
    writers = {name: cv2.VideoWriter(str(dest/f"{name}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(960,540))
               for name in ("original","contours","repaired")}
    writers["comparison"] = cv2.VideoWriter(str(dest/"comparison.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1920,396))
    if not all(w.isOpened() for w in writers.values()):
        raise RuntimeError("视频编码器无法打开")
    results, gif = [], []
    baseline_start = perf_counter()
    try:
        for i in range(frames):
            original, clean, truth, reference = sample_frame(i)
            x,y,w,h = 76,324,808,194
            result = processor.process(original[y:y+h,x:x+w], force_detection=(i % 6 == 0))
            repaired = original.copy()
            repaired[y:y+h,x:x+w] = result.repaired
            mask = np.zeros(original.shape[:2],np.uint8)
            mask[y:y+h,x:x+w] = result.mask
            contours = original.copy()
            contours[y:y+h,x:x+w] = result.debug_image()
            changed = np.any(repaired != original,axis=2)
            truth_bool = truth > 0
            english = polygon_mask(original.shape[:2], [r for r in reference if not r.chinese],2)>0
            intersection = np.count_nonzero((mask>0)&truth_bool)
            truth_count, predicted = np.count_nonzero(truth_bool),np.count_nonzero(mask)
            error_before = float(np.mean(np.abs(original[truth_bool].astype(float)-clean[truth_bool]))) if truth_count else 0
            error_after = float(np.mean(np.abs(repaired[truth_bool].astype(float)-clean[truth_bool]))) if truth_count else 0
            results.append({"frame":i,"scene":SCENES[(i//48)%6],**result.timing,
                            "chinese_glyph_recall":intersection/max(1,truth_count),
                            "mask_precision":intersection/max(1,predicted),
                            "english_changed_pixels":int(np.count_nonzero(changed&english)),
                            "outside_mask_changed_pixels":int(np.count_nonzero(changed&(mask==0))),
                            "blank_frame_mask_pixels":predicted if truth_count==0 else 0,
                            "truth_pixels":truth_count,"mask_pixels":predicted,
                            "before_mae":error_before,"after_mae":error_after,
                            "status":result.status,
                            "recognized":[r.text for r in result.regions]})
            for key,image in (("original",original),("contours",contours),("repaired",repaired)):
                writers[key].write(image)
            comparison = np.concatenate((label(original,"A / ORIGINAL"),label(contours,"B / GLYPH MASK + ENGLISH GUARD"),
                                         label(repaired,"C / OPENCV LOCAL REPAIR")),axis=1)
            writers["comparison"].write(comparison)
            if i % 4 == 0:
                gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(comparison,(1200,248)),cv2.COLOR_BGR2RGB)))
            if i % 48 == 12:
                write_image(dest/f"scene-{i//48+1}.png", comparison)
            # 3 one-second ROI clips: simple background, face illustration, motion.
            for slug,start in (("simple",0),("face",96),("motion",144)):
                if start <= i < start+24:
                    # Keep useful surrounding context, dimensions divisible by 8.
                    crop = original[296:528,64:896]
                    clip_mask = mask[296:528,64:896]
                    clip_clean = clean[296:528,64:896]
                    write_image(dest/"research_inputs"/slug/"frames"/f"{i-start:04}.png",cv2.resize(crop,(576,160)))
                    write_image(dest/"research_inputs"/slug/"masks"/f"{i-start:04}.png",cv2.resize(clip_mask,(576,160),interpolation=cv2.INTER_NEAREST))
                    write_image(dest/"research_inputs"/slug/"clean"/f"{i-start:04}.png",cv2.resize(clip_clean,(576,160)))
            if i % 48 == 0:
                print(f"Validated {i}/{frames} frames",flush=True)
    finally:
        for writer in writers.values():
            writer.release()
    for name in writers:
        make_browser_video(dest/f"{name}.mp4")
    if gif:
        gif[0].save(dest/"comparison.gif",save_all=True,append_images=gif[1:],duration=167,loop=0,optimize=True)
    metrics = {"created":datetime.now().isoformat(timespec="seconds"),"source":"synthetic, with known clean backgrounds",
               "frames":frames,"fps":fps,"resolution":[960,540],"roi":[76,324,808,194],
               "elapsed_including_generation_s":perf_counter()-baseline_start,
               "processing_ms":summary([r["total_ms"] for r in results]),
               "ocr_processing_ms":summary([r["total_ms"] for r in results if r["ocr_ms"]>0]),
               "cached_processing_ms":summary([r["total_ms"] for r in results if r["ocr_ms"]==0]) if any(r["ocr_ms"]==0 for r in results) else None,
               "english_changed_pixels":sum(r["english_changed_pixels"] for r in results),
               "outside_mask_changed_pixels":sum(r["outside_mask_changed_pixels"] for r in results),
               "blank_frame_mask_pixels":sum(r["blank_frame_mask_pixels"] for r in results),
               "by_scene":[]}
    for scene in SCENES:
        group = [r for r in results if r["scene"]==scene]
        visible = [r for r in group if r["truth_pixels"]]
        if group:
            metrics["by_scene"].append({"scene":scene,"frames":len(group),
                "processing_ms":summary([r["total_ms"] for r in group]),
                "glyph_recall":float(np.mean([r["chinese_glyph_recall"] for r in visible])) if visible else 0,
                "before_mae":float(np.mean([r["before_mae"] for r in visible])) if visible else 0,
                "after_mae":float(np.mean([r["after_mae"] for r in visible])) if visible else 0,
                "english_changed_pixels":sum(r["english_changed_pixels"] for r in group),
                "blank_frame_mask_pixels":sum(r["blank_frame_mask_pixels"] for r in group)})
    (dest/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=json_value),encoding="utf-8")
    (dest/"frames.json").write_text(json.dumps(results,ensure_ascii=False,indent=2,default=json_value),encoding="utf-8")
    write_report(dest)
    print(json.dumps(metrics,ensure_ascii=False,indent=2,default=json_value),flush=True)
    return metrics


def write_report(dest: Path):
    metrics = json.loads((dest/"metrics.json").read_text(encoding="utf-8"))
    research_path = dest/"propainter_metrics.json"
    research = json.loads(research_path.read_text(encoding="utf-8")) if research_path.exists() else None
    live_path = dest.parent/"live-smoke"/"async_metrics.json"
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else None
    live_section = ""
    if live:
        live_section = f"""<section><h2>独立窗口实时采集：异步识别路线</h2><div class="metrics">
<div><strong>{live['steady_processed_fps']:.1f} fps</strong>稳定阶段处理更新</div>
<div><strong>{live['first_repair_after_s']:.1f} s</strong>首次显示修补前的等待</div>
<div><strong>{live['processing_ms_p50']:.1f} ms</strong>逐帧处理 P50</div>
<div><strong>{live['callback_to_completed_ms_p50']:.1f} ms</strong>回调至处理完成 P50</div></div>
<p>使用另一个进程的合成播放器窗口，桌面缩放为 150%。背景识别与逐帧修补分开，识别期间保留原图。
逐帧处理 P95 为 {live['processing_ms_p95']:.1f} ms；回调至完成 P95 为 {live['callback_to_completed_ms_p95']:.1f} ms。
这里未计入界面呈现和屏幕扫描，且只测试了固定字幕布局。手动桌面交互检查中途停止，尚未完成真实 B 站视频的交互验收。</p></section>"""
    rows = "".join(f"<tr><td>{html.escape(s['scene'])}</td><td>{s['glyph_recall']:.1%}</td>"
                   f"<td>{s['before_mae']:.1f} → {s['after_mae']:.1f}</td>"
                   f"<td>{s['processing_ms']['p50']:.1f} / {s['processing_ms']['p95']:.1f}</td>"
                   f"<td>{s['english_changed_pixels']}</td><td>{s['blank_frame_mask_pixels']}</td></tr>"
                   for s in metrics["by_scene"])
    research_section = "<p>尚未运行。不会将离线模型的能力描述为实时去字幕能力。</p>"
    if research:
        research_section = "<p>本次三段合成样例中，ProPainter 在纯色和运动片段降低了误差，在人脸示意片段误差更高。它没有稳定优于轻量修补，当前保留 OpenCV 用于实时原型，AI 模型用于离线效果研究。</p>"
        clip_names = {"simple":"纯色背景","face":"人脸示意","motion":"运动背景"}
        for clip in research.get("clips",[]):
            research_section += f"<h3>{clip_names.get(clip['clip'],html.escape(clip['clip']))}</h3>"
            if clip.get("return_code")==0:
                research_section += f"""<p>处理 {clip['frames']} 帧（1 秒视频，576 × 160）共用 {clip['elapsed_s']:.1f} 秒，包含模型启动和编码。
本进程分配显存峰值 {clip.get('torch_peak_allocated_mib',0):.0f} MiB。</p>
<p>修补区域内背景误差：轻量修补 <strong>{clip['opencv_mae_inside_mask']:.2f}</strong>，
ProPainter <strong>{clip['propainter_mae_inside_mask']:.2f}</strong>（越低越好）。</p>"""
            else:
                research_section += "<p>该片段运行失败，详见完整测量记录。</p>"
            if clip.get("comparison"):
                research_section += f'<video controls loop muted src="{html.escape(clip["comparison"])}"></video>'
        research_section += '<p><a href="propainter_metrics.json">查看完整测量记录</a></p>'
    processing = metrics["processing_ms"]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>留英文 · 去字幕验证报告</title><style>
body{{margin:0;background:#f2f5fa;color:#20324d;font:15px/1.75 'Microsoft YaHei',sans-serif}}main{{max-width:1180px;margin:44px auto;padding:0 24px}}h1{{font-size:38px;margin-bottom:6px}}.muted{{color:#6d7f95}}.tag{{color:#21775f;background:#e1f3eb;padding:6px 12px;border-radius:20px}}section{{background:white;border:1px solid #e0e7f0;border-radius:18px;padding:26px;margin:22px 0}}video,img{{width:100%;border-radius:12px;background:#152033}}.metrics{{display:flex;gap:24px;flex-wrap:wrap}}.metrics strong{{display:block;font-size:29px;color:#3757cf}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px 8px;text-align:left;border-bottom:1px solid #e7ecf3}}pre{{white-space:pre-wrap;background:#f4f6fa;padding:16px;border-radius:10px}}a{{color:#3757cf}}.scroll{{overflow:auto}}
</style><main><span class="tag">本机实测 / 合成测试素材</span><h1>去掉中文，画面保留了多少？</h1>
<p class="muted">{html.escape(metrics['created'])} · 六种场景 · {metrics['frames']} 帧 · 960 × 540</p>
{live_section}<section><h2>六场景离线质量验证：同步识别基线</h2><div class="metrics"><div><strong>{processing['p50']:.1f} ms</strong>处理耗时中位数</div><div><strong>{processing['p95']:.1f} ms</strong>处理耗时 P95</div><div><strong>{metrics['english_changed_pixels']}</strong>英文区域改动像素</div><div><strong>{metrics['outside_mask_changed_pixels']}</strong>轮廓以外改动像素</div></div>
<p>耗时包含本地识别、文字轮廓提取和 OpenCV 修补，不包含屏幕采集及显示延迟。原型界面另行显示“回调至显示”，该数字也不等于真实端到端光学延迟。轮廓之外的像素保留，不代表轮廓一定识别正确。</p></section>
<section><h2>动态对比：原图 / 文字轮廓 / 局部修补</h2><video controls autoplay muted loop src="comparison.mp4"></video>
<p>视频按 24 fps 回放，并非实时处理录像。绿色为实际修补区域；蓝色框为英文保护区域。素材为生成的测试画面，人脸为示意插画，不能据此宣称真实人脸、任意视频或字体都能可靠处理。</p>
<p><a href="comparison.gif">查看 GIF</a> · <a href="original.mp4">原画面</a> · <a href="contours.mp4">文字轮廓</a> · <a href="repaired.mp4">修补结果</a> · <a href="metrics.json">完整指标</a></p></section>
<section><h2>逐场景检查</h2><div class="scroll"><table><thead><tr><th>场景</th><th>中文字形覆盖率</th><th>已知背景误差 ↓</th><th>耗时 P50 / P95 ms</th><th>英文改动</th><th>空字幕帧误修补像素</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>字形覆盖率包含描边；误差为真实中文遮挡位置对照已知干净背景的平均绝对误差（0–255，越低越好）。细小文字仍可有残留，覆盖率不能代替人工观看判断。</p></section>
<section><h2>视频背景修补：ProPainter 离线对比</h2>{research_section}<p>ProPainter 在独立环境、短片段上运行；结果用于研究效果与成本，并未接入实时覆盖层。</p></section>
<section><h2>当前边界</h2><ul><li>优先支持中英文分行、白色或黄色、深色描边字幕；混排、低置信度区域保留原图。</li><li>复杂背景可能残留文字、产生模糊；实时叠加有处理延迟，不能保证零露字。</li><li>播放器全屏、缩放、字幕排版变化后重新框选。受保护视频可能无法捕获。</li><li>缺少实际网站及客户端视频样本；本报告不替代 B 站等真实场景兼容性测试。</li></ul></section></main></html>"""
    (dest/"report.html").write_text(page,encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_report(args.output)
    else:
        run_validation(args.output,args.seconds)
