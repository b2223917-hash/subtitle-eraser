import json
from pathlib import Path
import sys
from time import perf_counter,sleep
import numpy as np
from PySide6.QtCore import QCoreApplication
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from subtitle_eraser.windows import list_windows
from subtitle_eraser.worker import ProcessingWorker

app=QCoreApplication([])
windows=[w for w in list_windows() if w.title=="留英文 · 捕获测试播放器"]
assert len(windows)==1
worker=ProcessingWorker(window=windows[0])
worker.configure((.10,.63,.80,.32),worker.settings)
worker.start()
started=perf_counter()
last=-1
records=[]
first_repair=None
while perf_counter()-started<12:
    _,packet=worker.snapshot()
    if packet and packet.sequence!=last:
        last=packet.sequence
        elapsed=perf_counter()-started
        has_mask=bool(np.any(packet.result.mask))
        if has_mask and first_repair is None:
            first_repair=elapsed
        records.append({"elapsed_s":elapsed,"has_mask":has_mask,**packet.result.timing,
                        "callback_to_completed_ms":(packet.completed_at-packet.captured_at)*1000})
    sleep(.01)
worker.stop()
assert worker.wait(5000)
assert first_repair is not None,"No successful layout recognition"
active=[r for r in records if r["has_mask"]]
steady=[r for r in records if r["elapsed_s"]>first_repair+1]
metrics={"source":"dedicated synthetic WGC player, 150% desktop scaling",
         "frames_observed":len(records),"first_repair_after_s":first_repair,
         "steady_processed_fps":len(steady)/(12-first_repair-1),
         "processing_ms_p50":float(np.median([r["total_ms"] for r in active])),
         "processing_ms_p95":float(np.percentile([r["total_ms"] for r in active],95)),
         "callback_to_completed_ms_p50":float(np.median([r["callback_to_completed_ms"] for r in active])),
         "callback_to_completed_ms_p95":float(np.percentile([r["callback_to_completed_ms"] for r in active],95)),
         "background_ocr_ms_max":max(r["background_ocr_ms"] for r in records),
         "note":"Capture callback to completed excludes UI presentation. OCR runs asynchronously; quality report separately uses synchronous detection."}
dest=Path("outputs/live-smoke/async_metrics.json")
dest.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
