import logging
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from cassava.transforms import get_test_transforms
from cassava.utils import DatasetFromSubset
from matplotlib import pyplot as plt


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
                                         shuffle=False,
                                         drop_last=False)

    predictions = []
    probas = []
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            if torch.cuda.is_available():
                images = images.cuda()
            batch_probas = model.predict_proba(images)
            batch_preds = torch.max(batch_probas, 1)[1]
            predictions.append(batch_preds)
            probas.append(batch_probas)

    predictions = torch.hstack(predictions).flatten().tolist()
    probas = torch.vstack(probas).tolist()

    return predictions, probas


def lr_find(trainer, model, train_data_loader, val_data_loader=None, plot=False):
    val_dataloaders = [val_data_loader] if  val_data_loader else None
    lr_finder = trainer.tuner.lr_find(model,
                                      train_dataloader=train_data_loader,
                                      val_dataloaders=val_dataloaders)
    if plot:
        plt.figure()
        plt.title('LR finder results')
        lr_finder.plot(suggest=True)
        plt.show()

    newlr = lr_finder.suggestion()
    logging.info('LR finder suggestion: %f', newlr)

    return newlr
