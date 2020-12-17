import logging
from argparse import Namespace
from sklearn.model_selection import train_test_split
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cassava.models.model import LeafDoctorModel
from cassava.models.byol import BYOL
from cassava.transforms import get_train_transforms, get_test_transforms
from cassava.utils import DatasetFromSubset
from cassava.pipelines.predict.nodes import predict
from cassava.node_helpers import score


def split_data(train_labels, parameters):
    """Splits trainig data into the train and validation set"""
    train_indices, val_indices = train_test_split(range(len(train_labels)),
                     stratify=train_labels.label,
                     random_state=parameters['seed'],
                     test_size=parameters['validation_size'])
    return train_indices, val_indices


def train_model(finetuned_model, train_images_lmdb, train_indices, val_indices, parameters):
    train_transform, val_transform = get_train_transforms(), get_test_transforms()

    train_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_lmdb, indices=train_indices),
                                      transform=train_transform)

    val_dataset = DatasetFromSubset(torch.utils.data.Subset(train_images_lmdb, indices=val_indices),
                                    transform=val_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=parameters['classifier']['batch_size'],
                                                    num_workers=parameters['data_loader_workers'],
                                                    shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  num_workers=parameters['data_loader_workers'],
                                                  batch_size=parameters['classifier']['batch_size'])

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_acc",
                                       mode='max',
                                       verbose=True,
                                       dirpath=parameters['classifier']['checkpoints_dir'],
                                       filename="{epoch}_{val_acc:.4f}",
                                       save_top_k=parameters['classifier']['save_top_k_checkpoints'])
    early_stopping = EarlyStopping('val_acc',
                                   mode='max',
                                   patience=parameters['classifier']['early_stop_patience'],
                                   verbose=True,
                                   )

    hparams = Namespace(**parameters['classifier'])

    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch = True,
        terminate_on_nan=True,
        callbacks=[model_checkpoint, early_stopping],
    )

    # Model
    model = LeafDoctorModel(hparams)
    model.load_state_dict(finetuned_model.state_dict())

    # Training
    trainer.fit(model, train_data_loader, val_data_loader)
    logging.info('Training finished')

    # Saving
    best_checkpoint = model_checkpoint.best_model_path
    model = LeafDoctorModel().load_from_checkpoint(checkpoint_path=best_checkpoint)
    return model


def score_model(model, train_images_torch, indices, parameters):
    logging.info('Scoring model')
    if parameters['classifier'].get('limit_val_batches'):
        indices = indices[:parameters['classifier']['limit_val_batches']*parameters['classifier']['batch_size']]
    labels = train_images_torch.labels[indices]
    predictions, probas = predict(model,
                          dataset=train_images_torch,
                          indices=indices,
                          batch_size=parameters['classifier']['batch_size'],
                          num_workers=parameters['data_loader_workers'],
                          transform=get_test_transforms())

    scores = score(predictions, labels)

    logging.info(f'Validation scores:\n{scores}')
    return scores, predictions
