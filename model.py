import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFeatureEncoder(nn.Module):
    """Pool spatial features into one vector per frame (baseline)."""

    def __init__(self, feature_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        # features: [B, T, 512, 7, 7]
        if features.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(features.shape)}"
            )

        batch_size, num_frames, channels, height, width = features.shape

        if channels != 512:
            raise ValueError(f"Expected 512 channels, got {channels}")

        # Average spatial dimensions: [B,T,512,7,7] -> [B,T,512]
        features = features.mean(dim=(-2, -1))

        encoded = self.projection(features)
        encoded = self.layer_norm(encoded)
        encoded = F.relu(encoded)
        encoded = self.dropout(encoded)

        return encoded


class TemporalBiGRUFeatureEncoder(nn.Module):
    """Dual spatial pooling (Mean + Max) with temporal BiGRU modeling."""

    def __init__(self, feature_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        # Concat mean pool (512) and max pool (512) -> 1024 -> hidden_dim
        self.spatial_projection = nn.Linear(feature_dim * 2, hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_dropout = nn.Dropout(dropout)

        # Bidirectional GRU across the temporal dimension (T=8 frames)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_dropout = nn.Dropout(dropout)

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(features.shape)}"
            )

        batch_size, num_frames, channels, height, width = features.shape
        if channels != 512:
            raise ValueError(f"Expected 512 channels, got {channels}")

        # 1. Dual Spatial Pooling (preserves salient high-activation localized objects)
        mean_spatial = features.mean(dim=(-2, -1))  # [B, T, 512]
        max_spatial = features.amax(dim=(-2, -1))   # [B, T, 512]
        dual_spatial = torch.cat([mean_spatial, max_spatial], dim=-1)  # [B, T, 1024]

        projected = self.spatial_projection(dual_spatial)
        projected = self.spatial_norm(projected)
        projected = F.relu(projected)
        projected = self.spatial_dropout(projected)  # [B, T, 512]

        # 2. Temporal modeling across frames with residual connection
        gru_out, _ = self.temporal_gru(projected)  # [B, T, 512]
        encoded = self.temporal_norm(projected + gru_out)
        encoded = self.temporal_dropout(encoded)  # [B, T, 512]

        return encoded


# Keep SpatioTemporalFeatureEncoder as backwards-compatible alias
SpatioTemporalFeatureEncoder = TemporalBiGRUFeatureEncoder


class SpatioTemporalAttentionFeatureEncoder(nn.Module):
    """Projects all 8x49=392 region vectors with learned frame and spatial position embeddings."""

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=512,
        num_frames=8,
        num_spatial=49,
        dropout=0.3,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_spatial = num_spatial
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.frame_embed = nn.Embedding(num_frames, hidden_dim)
        self.spatial_embed = nn.Embedding(num_spatial, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        # features: [B, T, C, H, W]
        if features.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(features.shape)}"
            )

        batch_size, num_frames, channels, height, width = features.shape
        if channels != 512:
            raise ValueError(f"Expected 512 channels, got {channels}")
        if num_frames != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {num_frames}")
        if height * width != self.num_spatial:
            raise ValueError(
                f"Expected {self.num_spatial} spatial positions, got {height * width}"
            )

        # [B, T, C, H, W] -> [B, T, H*W, C]
        x = features.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_frames, self.num_spatial, channels
        )
        # Project 512 -> hidden_dim
        projected = self.projection(x)  # [B, T, H*W, hidden_dim]

        # Learned frame-index and spatial-position embeddings
        frame_idx = torch.arange(num_frames, device=features.device)
        spatial_idx = torch.arange(self.num_spatial, device=features.device)

        frame_emb = self.frame_embed(frame_idx).view(1, num_frames, 1, -1)
        spatial_emb = self.spatial_embed(spatial_idx).view(1, 1, self.num_spatial, -1)

        out = projected + frame_emb + spatial_emb
        out = out.reshape(batch_size, num_frames * self.num_spatial, -1)  # [B, 392, hidden_dim]
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


def reshape_attention_weights(weights, num_frames=8, height=7, width=7):
    """Reshape attention weights [B, L, 392] to [B, L, 8, 7, 7]."""
    expected = num_frames * height * width
    if weights.shape[-1] != expected:
        raise ValueError(
            f"Expected last dimension to be {expected}, got {weights.shape[-1]}"
        )
    return weights.view(*weights.shape[:-1], num_frames, height, width)


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention generalized across N positions."""

    def __init__(self, feature_dim=512, hidden_dim=512, attention_dim=512):
        super().__init__()
        self.W1 = nn.Linear(feature_dim, attention_dim)
        self.W2 = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, hidden_state, precomputed_W1=None):
        # encoder_out: [B, N, feature_dim]
        # hidden_state: [B, hidden_dim]
        # precomputed_W1: [B, N, attention_dim] (optional precomputed W1(encoder_out))
        if precomputed_W1 is None:
            pos_projection = self.W1(encoder_out)
        else:
            pos_projection = precomputed_W1

        hidden_projection = self.W2(hidden_state).unsqueeze(1)

        energy = torch.tanh(pos_projection + hidden_projection)
        scores = self.V(energy).squeeze(-1)  # [B, N]

        attention_weights = F.softmax(scores, dim=1)  # [B, N]

        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_out
        ).squeeze(1)  # [B, feature_dim]

        return context_vector, attention_weights


# Keep AdditiveTemporalAttention as backwards-compatible alias
AdditiveTemporalAttention = AdditiveAttention


class CaptionDecoder(nn.Module):
    """LSTMCell decoder with additive attention and optional scheduled sampling."""

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        feature_dim=512,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.attention = AdditiveAttention(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            attention_dim=hidden_dim,
        )

        self.lstm_cell = nn.LSTMCell(
            embed_dim + feature_dim, hidden_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_out, captions, scheduled_sampling_prob=0.0):
        # encoder_out: [B, N, hidden_dim]
        # captions: [B, L]
        embeddings = self.embedding(captions)

        # Initialize hidden state from mean representation across all positions.
        hidden = encoder_out.mean(dim=1)
        cell = torch.zeros_like(hidden)

        # Precompute key projection W1(encoder_out) once per sequence
        precomputed_W1 = self.attention.W1(encoder_out)

        outputs = []
        attention_history = []
        prev_pred_embed = None

        for t in range(embeddings.size(1)):
            context, attention_weights = self.attention(
                encoder_out, hidden, precomputed_W1=precomputed_W1
            )

            # Scheduled sampling during training
            if (
                t > 0
                and self.training
                and scheduled_sampling_prob > 0.0
                and torch.rand(1).item() < scheduled_sampling_prob
                and prev_pred_embed is not None
            ):
                step_embed = prev_pred_embed
            else:
                step_embed = embeddings[:, t, :]

            lstm_input = torch.cat(
                [step_embed, context], dim=1
            )

            hidden, cell = self.lstm_cell(
                lstm_input, (hidden, cell)
            )

            logits = self.output_projection(self.dropout(hidden))

            outputs.append(logits)
            attention_history.append(attention_weights)

            if scheduled_sampling_prob > 0.0 and self.training:
                pred_tokens = logits.argmax(dim=-1)
                prev_pred_embed = self.embedding(pred_tokens)

        # logits: [B, L, V]; attention: [B, L, N]
        return (
            torch.stack(outputs, dim=1),
            torch.stack(attention_history, dim=1),
        )


class VideoCaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        feature_dim=512,
        embed_dim=256,
        hidden_dim=512,
        dropout=0.3,
        encoder_type="temporal",
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type in ("spatiotemporal", "temporal_bigru"):
            self.encoder = TemporalBiGRUFeatureEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif encoder_type == "spatiotemporal_attention":
            self.encoder = SpatioTemporalAttentionFeatureEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif encoder_type == "temporal":
            self.encoder = TemporalFeatureEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, video_features, captions, scheduled_sampling_prob=0.0):
        encoded_features = self.encoder(video_features)

        logits, attention_weights = self.decoder(
            encoded_features, captions, scheduled_sampling_prob=scheduled_sampling_prob
        )

        return logits, attention_weights
