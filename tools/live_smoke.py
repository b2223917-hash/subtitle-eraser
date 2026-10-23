"""Read-only window-capture integration test against our dedicated test player."""
import json
from pathlib import Path
import sys
from time import perf_counter, sleep
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from subtitle_eraser.windows import list_windows
from subtitle_eraser.capture import WindowCapture
from subtitle_eraser.pipeline import SubtitleProcessor
from subtitle_eraser.validate import write_image

candidates=[w for w in list_windows() if w.title=="留英文 · 捕获测试播放器"]
if len(candidates)!=1:
    raise RuntimeError("Expected one dedicated test player")
window=candidates[0]
source=WindowCapture(window.hwnd,window.title)
source.start()
started=perf_counter()
sequences=set()
last=None
try:
    while perf_counter()-started<6:
        last=source.read()
        if last:
            sequences.add(last[0])
        sleep(.04)
finally:
    source.stop()
if last is None:
    raise RuntimeError("WGC did not deliver a frame")
sequence,image,captured_at=last
dest=Path("outputs/live-smoke")
dest.mkdir(parents=True,exist_ok=True)
write_image(dest/"captured.png",image)
height,width=image.shape[:2]
crop=image[int(height*.63):int(height*.95),int(width*.10):int(width*.90)]
result=SubtitleProcessor().process(crop,force_detection=True)
write_image(dest/"repaired.png",result.repaired)
data={"frames_observed":len(sequences),"last_sequence":sequence,"size":[width,height],
      "recognized":[r.text for r in result.regions],"timing":result.timing,
      "mask_pixels":int((result.mask>0).sum()),"status":result.status}
assert len(sequences)>2
assert any(r.chinese for r in result.regions)
assert any(not r.chinese for r in result.regions)
(dest/"capture_metrics.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(data,ensure_ascii=False,indent=2))
