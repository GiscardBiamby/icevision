__all__ = ["COCOMetric", "COCOMetricType"]

from typing import Dict

from icevision.data import *
from icevision.imports import *
from icevision.metrics.metric import *
from icevision.utils import *
from omegaconf import DictConfig
from pycocotools.cocoeval import Params, StatKey, StatKeyPerClass


class COCOMetricType(Enum):
    """Available options for `COCOMetric`."""

    bbox = "bbox"
    mask = "segm"
    keypoint = "keypoints"


class COCOMetric(Metric):
    """Wrapper around [cocoapi evaluator](https://github.com/cocodataset/cocoapi)

    Calculates average precision.

    # Arguments
        metric_type: Dependent on the task you're solving.
        print_summary: If `True`, prints a table with statistics.
        show_pbar: If `True` shows pbar when preparing the data for evaluation.
    """

    def __init__(
        self,
        config: DictConfig,
        metric_type: COCOMetricType = COCOMetricType.bbox,
        iou_thresholds: Optional[Sequence[float]] = None,
        print_summary: bool = False,
        show_pbar: bool = False,
        class2id: Dict = None,
        summary_ious: Optional[List[float]] = None,
    ):
        self.metric_type = metric_type
        self.iou_thresholds = iou_thresholds
        self.summary_ious = summary_ious
        self.print_summary = print_summary
        self.show_pbar = show_pbar
        self._records, self._preds = [], []
        self._class2id = cast(Dict, class2id or {})

    def _reset(self):
        self._records.clear()
        self._preds.clear()

    def accumulate(self, preds):
        for pred in preds:
            self._records.append(pred.ground_truth)
            self._preds.append(pred.pred)

    def finalize(self) -> Dict[str, float]:
        with CaptureStdout():
            coco_eval = create_coco_eval(
                records=self._records,
                preds=self._preds,
                metric_type=self.metric_type.value,
                class2id=self._class2id,
                iou_thresholds=self.iou_thresholds,
                show_pbar=self.show_pbar,
            )
            if self.summary_ious:
                coco_eval.params.summaryIous = self.summary_ious
            coco_eval.params.maxDets = [1, 10, 100]
            coco_eval.params.areaRng = [
                [0 ** 2, 1e5 ** 2],
                # [0 ** 2, 32 ** 2],
                # [32 ** 2, 96 ** 2],
                # [96 ** 2, 1e5 ** 2],
            ]
            coco_eval.params.areaRngLbl = ["all"]
            coco_eval.evaluate()
            coco_eval.accumulate()

        with CaptureStdout(propagate_stdout=self.print_summary):
            coco_eval.summarize()

        stats = coco_eval.stats
        per_class_stats = coco_eval.stats_dict_per_class
        logs = {k: v for k, v in coco_eval.stats_dict.items()}
        logs = {**logs, **per_class_stats}
        # Copy the the key that gets logged to the progress bar for "COCOMetric":
        # print("coco_eval.stats_dict.keys: ", coco_eval.stats_dict.keys())
        logs["AP (IoU=0.75) area=all"] = coco_eval.stats_dict[
            StatKey("AP", iou="0.75", area="all", max_dets=coco_eval.params.maxDets[-1])
        ]

        self._reset()
        return logs
