from threading import Event
from time import perf_counter, sleep
import numpy as np

from subtitle_eraser.pipeline import AsyncOcrDetector, TextRegion


def test_slow_ocr_does_not_block_and_reset_discards_old_layout():
    entered,release=Event(),Event()
    class Slow:
        def detect(self,image):
            entered.set()
            release.wait(2)
            return [TextRegion(np.array([[0,0],[10,0],[10,10],[0,10]]),"中文")]
    detector=AsyncOcrDetector()
    detector.engine=Slow()
    before=perf_counter()
    assert detector.detect(np.zeros((30,60,3),np.uint8))==[]
    assert perf_counter()-before<.2
    assert entered.wait(1)
    detector.reset()
    release.set()
    detector.thread.join(2)
    assert detector.regions==[]
    detector.close()
