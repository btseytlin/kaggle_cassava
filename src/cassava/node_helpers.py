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
    model.freeze()
    for images, labels in tqdm(loader):
        batch_preds = model.predict(images)
        predictions += batch_preds.tolist()
        probas += model.predict_proba(images).tolist()
    return predictions, probas
