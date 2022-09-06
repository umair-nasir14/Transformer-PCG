

from Datasets import get_mario_loaders
from transformer import ImageGPT


def reproduce(
    n_epochs=457,
    batch_size=64,
    log_dir="/home/munasir/image-gpt/igpt-pytorch/pytorch-generative",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    """Training script with defaults to reproduce results.
    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.
    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    import torch
    from torch import optim
    from torch.nn import functional as F
    from torch.optim import lr_scheduler

    from pytorch_generative import datasets
    from pytorch_generative import models
    from pytorch_generative import trainer

    train_loader, train_eval_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, train_eval_loader = get_mario_loaders(
            batch_size, dynamically_binarize=False
        )

    model = ImageGPT(
        in_channels=3,
        out_channels=3,
        in_size_0=14,
        in_size_1=200,
        n_transformer_blocks=5,
        n_attention_heads=8,
        n_embedding_channels=16,
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-3)#5e-3
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.999977)

    def loss_fn(x, _, preds):
        x = torch.tensor(x, dtype=torch.float32)
        batch_size = x.shape[0]
        x, preds = x.reshape((batch_size, -1)), preds.reshape((batch_size, -1))
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        return loss.sum(dim=1).mean()

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=train_eval_loader,
        lr_scheduler=scheduler,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    
    model_trainer.interleaved_train_and_eval(n_epochs)

    return model

import wandb

wandb.init(project="Transformer-Mario-PCG",config = {
                    "learning_rate": 5e-3,
                    "epochs": 100,
                    "batch_size": 1,
                    "in_size_0":14,
                    "in_size_1":200,
                    "n_transformer_blocks":5,
                    "n_attention_heads":8,
                    "n_embedding_channels":16,
                    "Images": 1000,
                    "cnn_layers": 1
                    })

model = reproduce(n_epochs=1000,batch_size=1)

