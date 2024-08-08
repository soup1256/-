# main.py
import os
import torch
from torch.utils.data import DataLoader
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

def main():
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    print("Random seed set.")

    # Initialize the checkpoint utility
    checkpoint = utility.Checkpoint(args)
    print("Checkpoint initialized.")

    if checkpoint.ok:
        # Load the dataset
        print("Loading dataset...")
        loader = data.Data(args)
        print("Dataset loaded.")

        # Initialize the model
        print("Initializing model...")
        my_model = model.Model(args, checkpoint).to(args.device_id)
        if args.n_GPUs > 1:
            my_model = torch.nn.DataParallel(my_model)  # Use DataParallel for multi-GPU support
        print("Model initialized.")

        # Initialize the loss function if not in test-only mode
        my_loss = loss.Loss(args, checkpoint) if not args.test_only else None
        if my_loss:
            print("Loss function initialized.")

        # Initialize the trainer
        print("Initializing trainer...")
        trainer = Trainer(args, loader, my_model, my_loss, checkpoint)
        print("Trainer initialized.")

        # Run the training and testing loop
        while not trainer.terminate():
            trainer.train()
            trainer.test()

    # Mark the checkpoint as done
    checkpoint.done()
    print("Process completed.")

if __name__ == '__main__':
    main()
