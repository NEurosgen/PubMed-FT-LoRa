from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, dataset):
        preds, targets = [], []
        for row in dataset:
            pred = self.model.predict(row['question'])
            target = row['answer'].lower().strip()
            preds.append(pred)
            targets.append(target)
        report = classification_report(targets, preds, labels=["yes", "no", "maybe"], output_dict=True)
       
        return report