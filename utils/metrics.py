import numpy as np

class Metrics:
    def __init__(self, pred, mask, num_classes, ignore_index=None):
        """
        pred: Tensor of shape [B, H, W] (predicted class indices)
        mask: Tensor of shape [B, H, W] (ground truth class indices)
        num_classes: int (number of classes)
        ignore_index: int or None (class index to ignore in evaluation)
        """
        assert pred.shape == mask.shape, "Prediction and mask must have the same shape"

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.pred = pred
        self.mask = mask
        self.confusion_matrix = self._compute_confusion_matrix(pred, mask)

    def _compute_confusion_matrix(self, pred, mask):
        if self.ignore_index is not None:
            valid = (mask>=0) & (mask<self.num_classes) & (mask != self.ignore_index)
        else:
            valid = (mask>=0) & (mask<self.num_classes)
        confusion = np.bincount(
            self.num_classes * mask[valid].astype(int) + pred[valid].astype(int),
            minlength = self.num_classes ** 2
        ).astype(np.int64).reshape(self.num_classes, self.num_classes)
        return confusion # [num_classes, num_classes]
    
    #---------------- IoU ----------------#
    def per_class_iou(self):
        cm = self.confusion_matrix
        ious=[]
        for i in range(self.num_classes):
            tp = cm[i,i]
            fp = cm[:,i].sum()-tp
            fn = cm[i,:].sum()-tp
            denom = tp+fp+fn
            if denom == 0:
                iou = float('nan')
            else:
                iou = tp/denom
            ious.append(iou)
        return np.array(ious) # [num_classes]   
    def mean_iou(self):
        return np.nanmean(self.per_class_iou())
    
    #Based on mIoU, consider the weighted frequency of each class
    def frequency_weighted_iou(self):
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum() # shape: [num_classes]
        ious = self.per_class_iou() # shape: [num_classes]
        fw_iou = (freq[freq > 0] * ious[freq > 0]).sum()
        return fw_iou # scalar

    #---------------- Accuracy ----------------#
    #each class accuracy, same as per_class_recall
    def per_class_accuracy(self):
        cm = self.confusion_matrix
        accuracies = []
        for i in range(self.num_classes):
            tp = cm[i,i]
            denom = cm[i,:].sum()
            if denom == 0:
                acc = float('nan')
            else:
                acc = tp/denom
            accuracies.append(acc)
        return np.array(accuracies) # [num_classes]
    def mean_accuracy(self):
        return np.nanmean(self.per_class_accuracy())
    
    #micro accuracy、micro precision and micro recall are equal to overall accuracy
    def overall_accuracy(self):
        cm = self.confusion_matrix
        correct = np.diag(cm).sum()
        total = cm.sum()
        if total == 0:
            return float('nan')
        else:
            return correct/total
    
    #---------------- Precision ----------------#
    def per_class_precision(self):
        cm = self.confusion_matrix
        precisions = []
        for i in range(self.num_classes):
            tp = cm[i,i]
            denom = cm[:,i].sum()
            if denom ==0:
                prec = float('nan')
            else:
                prec = tp/denom
            precisions.append(prec)
        return np.array(precisions) # [num_classes]
    
    def precision(self,mode="macro"):
        if mode == "macro":
            return np.nanmean(self.per_class_precision())
        elif mode == "micro":
            return self.overall_accuracy()
        else:
            raise ValueError("mode should be 'macro' or 'micro'")
    
    #---------------- Recall ----------------#
    def per_class_recall(self):
        cm = self.confusion_matrix
        recalls = []
        for i in range(self.num_classes):
            tp = cm[i,i]
            denom = cm[i,:].sum()
            if denom == 0:
                rec = float('nan')
            else:
                rec = tp/denom
            recalls.append(rec)
        return np.array(recalls) # [num_classes]
    
    def recall(self,mode="macro"):
        if mode == "macro":
            return np.nanmean(self.per_class_recall())
        elif mode == "micro":
            return self.overall_accuracy()
        else:
            raise ValueError("mode should be 'macro' or 'micro'")
    
    #---------------- F1 Score ----------------#
    def per_class_f1(self):
        precisions = self.per_class_precision()
        recalls = self.per_class_recall()
        f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-6) #shape: [num_classes]
        return f1s
    
    def f1(self,mode="macro"):
        if mode == "macro":
            return np.nanmean(self.per_class_f1())
        elif mode == "micro":
            return self.overall_accuracy()
        else:
            raise ValueError("mode should be 'macro' or 'micro'")