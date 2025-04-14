import argparse
import torch
from torch import nn
from torch.optim import Adam

from .models import MODEL_FACTORY, save_model
from .datasets import road_dataset


def train():
    parser = argparse.ArgumentParser(description="Train a driving planner model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="Model type to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="drive_data",
        help="Path to the dataset directory (e.g., drive_data).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--transform_pipeline",
        type=str,
        default=None,
        help="Name of the transformation pipeline.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu or cuda)."
    )

    args = parser.parse_args()

    # Set transformation pipeline based on model type if not provided
    if args.transform_pipeline is None:
        if args.model == "cnn_planner":
            args.transform_pipeline = "default"
        else:
            args.transform_pipeline = "state_only"

    # Set the device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Instantiate the model and move it to the chosen device
    model = MODEL_FACTORY[args.model]()
    model.to(device)

    # Create an optimizer and loss function.
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss(reduction="none")  # we'll apply masking manually

    # Load training data using the provided load_data function.
    # This returns a DataLoader given that return_dataloader is True.
    train_loader = road_dataset.load_data(
        args.dataset,
        transform_pipeline=args.transform_pipeline,
        return_dataloader=True,
        batch_size=args.batch_size,
        shuffle=True,
    )

    num_epochs = args.epochs
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # For mlp and transformer models, use lane boundaries; for cnn, use images.
            if args.model in ["mlp_planner", "transformer_planner"]:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device).float()  # (B, n_waypoints)
                pred = model(track_left, track_right)
            elif args.model == "cnn_planner":
                image = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device).float()  # (B, n_waypoints)
                pred = model(image)
            else:
                raise ValueError("Unsupported model type")

            # Compute the L1 loss (mean absolute error) elementwise.
            loss_val = loss_fn(pred, waypoints)  # (B, n_waypoints, 2)
            # Expand the mask to cover both dimensions.
            mask_expanded = mask.unsqueeze(-1)  # becomes (B, n_waypoints, 1)
            loss_val = loss_val * mask_expanded
            # Average loss only over the valid points.
            valid = mask_expanded.sum()
            if valid > 0:
                loss_val = loss_val.sum() / valid
            else:
                loss_val = loss_val.mean()

            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

    # Save the trained model weights using the provided save_model function.
    model_path = save_model(model)
    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    train()
