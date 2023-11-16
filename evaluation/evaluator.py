import torch
from sklearn.metrics import confusion_matrix
import numpy as np

class Evaluator():
    def __init__(self, testloader, model):
        self.testloader = testloader
        self.model = model

    def calculate_confusion(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)
    
    def get_confusion(self):
        predictions, true_labels = self.calculate_confusion()
        return confusion_matrix(true_labels, predictions)