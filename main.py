import os
import torch
from torch.utils.data import DataLoader
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

# Set the random seed for reproducibility
torch.manual_seed(args.seed)

# Initialize the checkpoint utility
checkpoint = utility.Checkpoint(args)

if checkpoint.ok:
    # Load the dataset
    loader = data.Data(args)
    
    # Initialize the model
    model = model.Model(args, checkpoint).to('cuda')  # Force to use CUDA
    if args.n_GPUs > 1:
        model = torch.nn.DataParallel(model)  # Use DataParallel for multi-GPU support
    
    # Initialize the loss function if not in test-only mode
    loss_fn = loss.Loss(args, checkpoint) if not args.test_only else None
    
    # Initialize the trainer
    trainer = Trainer(args, loader, model, loss_fn, checkpoint)
    
    # Run the training and testing loop
    while not trainer.terminate():
        trainer.train()
        trainer.test()

# Mark the checkpoint as done
checkpoint.done()
