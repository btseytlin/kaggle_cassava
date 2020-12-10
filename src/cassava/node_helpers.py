import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from cassava.transforms import get_test_transforms
from cassava.utils import DatasetFromSubset


def score(predictions, labels):
    return {
        'accuracy': accuracy_score(predictions, labels),
        'f1_score': f1_score(predictions, labels, average='weighted'),
    }


def predict(model, dataset, indices, batch_size=10, num_workers=4, transform=None):
    transform = transform or get_test_transforms()
    dataset = DatasetFromSubset(
        torch.utils.data.Subset(dataset, indices=indices),
        transform=transform)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=False)

    predictions = []
    probas = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            probas = model.predict_proba(images, cuda=True)
            batch_preds = torch.max(probas, 1)[1]
            predictions.append(batch_preds)
            probas.append(probas)

    predictions = torch.hstack(predictions).flatten().tolist()
    probas = torch.hstack(probas).flatten().tolist()

    return predictions, probas
