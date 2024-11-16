"""
Enhanced Transformer model for cryptocurrency price prediction.
Includes multi-task learning and improved attention mechanisms.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional

class PositionalEncoding(layers.Layer):
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, d_model: int) -> tf.Tensor:
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def split_heads(self, inputs: tf.Tensor, batch_size: int) -> tf.Tensor:
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]

        # Linear layers and split heads
        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)

        # Scaled dot-product attention
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        attention_weights = self.dropout1(attention_weights, training=training)

        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        # Final linear layer and residual connection
        output = self.dense(output)
        output = self.dropout2(output, training=training)
        return self.layer_norm(inputs + output)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        attention_output = self.attention(inputs, training)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(attention_output + ffn_output)

class CryptoTransformer(Model):
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_transformer_blocks: int = 4,
        mlp_units: list = [128, 64],
        dropout: float = 0.1,
        mlp_dropout: float = 0.2,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model

        # Input layers and embedding
        self.feature_embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.dropout = layers.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ]

        # Global average pooling and MLP layers
        self.global_avg_pooling = layers.GlobalAveragePooling1D()

        # Price prediction branch
        self.price_mlp = []
        for dim in mlp_units:
            self.price_mlp.extend([
                layers.Dense(dim, activation="relu"),
                layers.Dropout(mlp_dropout)
            ])
        self.price_output = layers.Dense(1, name="price_output")

        # Direction prediction branch
        self.direction_mlp = []
        for dim in mlp_units:
            self.direction_mlp.extend([
                layers.Dense(dim, activation="relu"),
                layers.Dropout(mlp_dropout)
            ])
        self.direction_output = layers.Dense(1, activation="sigmoid", name="direction_output")

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        # Input embedding and positional encoding
        x = self.feature_embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)

        # Global pooling
        x = self.global_avg_pooling(x)

        # Price prediction branch
        price_features = x
        for layer in self.price_mlp:
            price_features = layer(price_features, training=training)
        price_output = self.price_output(price_features)

        # Direction prediction branch
        direction_features = x
        for layer in self.direction_mlp:
            direction_features = layer(direction_features, training=training)
        direction_output = self.direction_output(direction_features)

        return price_output, direction_output

def create_model(
    sequence_length: int,
    num_features: int,
    d_model: int = 128,
    num_heads: int = 8,
    ff_dim: int = 256,
    num_transformer_blocks: int = 4,
    mlp_units: list = [128, 64],
    dropout: float = 0.1,
    mlp_dropout: float = 0.2,
) -> CryptoTransformer:
    """
    Create a CryptoTransformer model instance with the specified parameters.
    """
    model = CryptoTransformer(
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=mlp_units,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
    )

    # Compile with custom loss weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "price_output": "mse",
            "direction_output": "binary_crossentropy"
        },
        loss_weights={
            "price_output": 1.0,
            "direction_output": 0.2
        },
        metrics={
            "price_output": ["mae", "mse"],
            "direction_output": ["accuracy"]
        }
    )

    return model
