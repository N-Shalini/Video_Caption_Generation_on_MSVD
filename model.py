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


class SpatioTemporalFeatureEncoder(nn.Module):
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


class AdditiveTemporalAttention(nn.Module):
    """Bahdanau additive attention across video frames."""

    def __init__(self, feature_dim=512, hidden_dim=512, attention_dim=512):
        super().__init__()
        self.W1 = nn.Linear(feature_dim, attention_dim)
        self.W2 = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, hidden_state):
        # encoder_out: [B,T,512]
        # hidden_state: [B,512]

        frame_projection = self.W1(encoder_out)
        hidden_projection = self.W2(hidden_state).unsqueeze(1)

        energy = torch.tanh(frame_projection + hidden_projection)
        scores = self.V(energy).squeeze(-1)  # [B,T]

        # Normalize attention across time (the frames).
        attention_weights = F.softmax(scores, dim=1)

        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_out
        ).squeeze(1)  # [B,512]

        return context_vector, attention_weights


class CaptionDecoder(nn.Module):
    """LSTMCell decoder with temporal additive attention and optional scheduled sampling."""

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

        self.attention = AdditiveTemporalAttention(
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
        # encoder_out: [B,T,512]
        # captions: [B,L]
        embeddings = self.embedding(captions)

        # Initialize hidden state from the mean frame representation.
        hidden = encoder_out.mean(dim=1)
        cell = torch.zeros_like(hidden)

        outputs = []
        attention_history = []
        prev_pred_embed = None

        for t in range(embeddings.size(1)):
            context, attention_weights = self.attention(
                encoder_out, hidden
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

        # logits: [B,L,V]; attention: [B,L,T]
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

        if encoder_type == "spatiotemporal":
            self.encoder = SpatioTemporalFeatureEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.encoder = TemporalFeatureEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

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
