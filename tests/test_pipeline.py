import cv2
import numpy as np

from subtitle_eraser.pipeline import (Settings, SubtitleProcessor, TextRegion,
                                      glyph_mask, polygon_mask, repair)
from subtitle_eraser.fixtures import sample_frame


class FixedDetector:
    def __init__(self, regions):
        self.regions = regions

    def detect(self, image):
        return self.regions


def test_english_and_pixels_outside_mask_are_preserved():
    image, clean, truth, regions = sample_frame(0)
    result = SubtitleProcessor(FixedDetector(regions)).process(image)
    english = polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3)>0
    assert np.array_equal(result.original[english],result.repaired[english])
    assert np.array_equal(result.original[result.mask==0],result.repaired[result.mask==0])
    assert np.count_nonzero(result.mask & truth)/np.count_nonzero(truth) > .85
    assert np.mean(np.abs(result.repaired[truth>0].astype(float)-clean[truth>0])) < np.mean(np.abs(image[truth>0].astype(float)-clean[truth>0]))


def test_no_subtitle_does_not_reuse_previous_pixels():
    image, _, _, regions = sample_frame(0)
    detector = FixedDetector(regions)
    processor = SubtitleProcessor(detector)
    processor.process(image,force_detection=True)
    blank = np.full_like(image,70)
    detector.regions = []
    result = processor.process(blank,force_detection=True)
    assert not np.any(result.mask)
    assert np.array_equal(blank,result.repaired)


def test_mixed_script_line_is_protected():
    image = np.full((120,400,3),30,np.uint8)
    cv2.putText(image,"English",(20,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    region = TextRegion(np.array([[10,20],[300,20],[300,90],[10,90]]),"中文 English")
    mask,status = glyph_mask(image,[region],Settings())
    assert not np.any(mask)
    assert "混排" in status


def test_ocr_failure_does_not_apply_old_mask():
    class BrokenDetector:
        def detect(self,image):
            raise RuntimeError("model unavailable")
    image,_,_,_ = sample_frame(0)
    result = SubtitleProcessor(BrokenDetector()).process(image)
    assert not np.any(result.mask)
    assert np.array_equal(image,result.repaired)
    assert "识别失败" in result.status


def test_english_guard_overrides_chinese():
    image = np.full((100,300,3),40,np.uint8)
    cv2.putText(image,"English",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    polygon=np.array([[10,10],[280,10],[280,90],[10,90]])
    mask,_ = glyph_mask(image,[TextRegion(polygon,"中文"),TextRegion(polygon,"English")],Settings())
    assert not np.any(mask)


def test_low_confidence_chinese_remains_untouched():
    image,_,_,regions = sample_frame(0)
    for r in regions:
        r.confidence=.1
    result=SubtitleProcessor(FixedDetector(regions)).process(image)
    assert not np.any(result.mask)
