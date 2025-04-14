from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Each of track_left and track_right is (n_track, 2) so concatenating
        # along track dimension gives (2*n_track, 2) and flattening produces a vector of length 4*n_track.
        input_dim = n_track * 2 * 2  # e.g., 10*2*2 = 40 for n_track=10
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waypoints * 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # track_left and track_right: (B, n_track, 2)
        # Concatenate along the track dimension -> (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        # Flatten to (B, 4*n_track)
        x = x.flatten(1)
        # Pass through MLP and reshape the output to (B, n_waypoints, 2)
        x = self.mlp(x)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for each waypoint (serving as the latent array)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        # Project the 2D coordinates into a d_model-dimensional embedding space.
        self.input_proj = nn.Linear(2, d_model)
        # Define a simple Transformer decoder using 2 layers and 4 attention heads.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # Project the transformer's output to 2D waypoint coordinates.
        self.out_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Combine lane boundary points from left and right.
        # Each is (B, n_track, 2) -> concatenated to (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        B = x.size(0)
        # Project 2D points into d_model-dimensional features.
        x = self.input_proj(x)  # (B, 2*n_track, d_model)
        # Transformer modules expect (S, B, E)
        x = x.transpose(0, 1)  # (2*n_track, B, d_model)
        # Prepare the query embeddings and repeat for the batch.
        queries = self.query_embed.weight  # (n_waypoints, d_model)
        queries = queries.unsqueeze(1).repeat(1, B, 1)  # (n_waypoints, B, d_model)
        # Run through the transformer decoder.
        out = self.decoder(tgt=queries, memory=x)  # (n_waypoints, B, d_model)
        out = out.transpose(0, 1)  # (B, n_waypoints, d_model)
        # Project to 2D output space.
        out = self.out_proj(out)  # (B, n_waypoints, 2)
        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer(
            "input_mean", torch.as_tensor(INPUT_MEAN), persistent=False
        )
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 16, 48, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 32, 24, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 12, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 6, 8)
            nn.ReLU(),
        )
        # After convolution, the features are flattened; here 128 * 6 * 8 = 6144.
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 8, 256), nn.ReLU(), nn.Linear(256, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[
            None, :, None, None
        ]

        x = self.cnn(x)
        # Flatten the feature map.
        x = x.view(x.size(0), -1)
        # Fully connected layers to produce waypoint predictions.
        x = self.fc(x)
        # Reshape to (B, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
