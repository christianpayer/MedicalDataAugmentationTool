
import numpy as np
import utils.sitk_np
import utils.sitk_image

class MetricBase(object):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        raise NotImplementedError()


class TpFpFnMetricBase(MetricBase):
    def calculate_tp_fp_fn(self, prediction_np, groundtruth_np, label):
        tp = np.sum(np.logical_and(prediction_np == label, groundtruth_np == label))
        fp = np.sum(np.logical_and(prediction_np == label, groundtruth_np != label))
        fn = np.sum(np.logical_and(prediction_np != label, groundtruth_np == label))
        return tp, fp, fn

    def calculate_tp_fp_fn_lists(self, prediction_sitk, groundtruth_sitk, labels):
        prediction_np = utils.sitk_np.sitk_to_np_no_copy(prediction_sitk)
        groundtruth_np = utils.sitk_np.sitk_to_np_no_copy(groundtruth_sitk)
        return [self.calculate_tp_fp_fn(prediction_np, groundtruth_np, label) for label in labels]

    def evaluate_function(self, tp, fp, fn):
        raise NotImplementedError()


class TpFpFnMetricPerLabel(TpFpFnMetricBase):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        tp_fp_fn_list = self.calculate_tp_fp_fn_lists(prediction_sitk, groundtruth_sitk, labels)
        return [self.evaluate_function(tp, fp, fn) for tp, fp, fn in tp_fp_fn_list]


class TpFpFnMetricGeneralized(TpFpFnMetricBase):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        tp_fp_fn_list = self.calculate_tp_fp_fn_lists(prediction_sitk, groundtruth_sitk, labels)
        tp, fp, fn = [sum(i) for i in zip(*tp_fp_fn_list)]
        return [self.evaluate_function(tp, fp, fn)]


class DiceMetric(TpFpFnMetricPerLabel):
    def evaluate_function(self, tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan


class JaccardMetric(TpFpFnMetricPerLabel):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fp + fn) if tp + fn > 0 else np.nan


class SpecificitySensitivityMetric(TpFpFnMetricPerLabel):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fn) if tp + fn > 0 else np.nan


class NegativePositivePredictiveValueMetric(TpFpFnMetricPerLabel):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fp) if tp + fn > 0 else np.nan


class GeneralizedDiceMetric(TpFpFnMetricGeneralized):
    def evaluate_function(self, tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan


class HausdorffDistanceMetric(MetricBase):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        return utils.sitk_image.hausdorff_distances(prediction_sitk, groundtruth_sitk, labels)

class SurfaceDistanceMetric(MetricBase):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        return utils.sitk_image.surface_distances(prediction_sitk, groundtruth_sitk, labels)


