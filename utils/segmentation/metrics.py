
import numpy as np
import utils.sitk_np

class MetricBase(object):
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        raise NotImplementedError()


class TpFpFnMetricBase(MetricBase):
    def calculate_tp_fp_fn(self, prediction_np, groundtruth_np, label):
        tp = np.sum(np.logical_and(prediction_np == label, groundtruth_np == label))
        fp = np.sum(np.logical_and(prediction_np == label, groundtruth_np != label))
        fn = np.sum(np.logical_and(prediction_np != label, groundtruth_np == label))
        return tp, fp, fn

    def calculate_tp_fp_fn_scores(self, prediction_sitk, groundtruth_sitk, labels):
        prediction_np = utils.sitk_np.sitk_to_np_no_copy(prediction_sitk)
        groundtruth_np = utils.sitk_np.sitk_to_np_no_copy(groundtruth_sitk)
        scores = []
        for label in labels:
            tp, fp, fn = self.calculate_tp_fp_fn(prediction_np, groundtruth_np, label)
            current_score = self.evaluate_function(tp, fp, fn)
            scores.append(current_score)
        return scores

    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        return self.calculate_tp_fp_fn_scores(prediction_sitk, groundtruth_sitk, labels)

    def evaluate_function(self, tp, fp, fn):
        raise NotImplementedError()


class DiceMetric(TpFpFnMetricBase):
    def evaluate_function(self, tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1


class JaccardMetric(TpFpFnMetricBase):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fp + fn) if tp + fp + fn > 0 else 1


class SpecificitySensitivityMetric(TpFpFnMetricBase):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fn) if tp + fn > 0 else 1


class NegativePositivePredictiveValueMetric(TpFpFnMetricBase):
    def evaluate_function(self, tp, fp, fn):
        return tp / (tp + fp) if tp + fp > 0 else 1
