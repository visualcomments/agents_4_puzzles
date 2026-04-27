import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .parallel import model_attr


class LegacyCompatibleEmbeddingBagLinear(nn.Module):
    """Exact replacement for Linear(one_hot(flat(indices))) with legacy checkpoint I/O."""

    def __init__(self, state_size, num_classes, out_features, bias=True):
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.out_features = int(out_features)
        self.in_features = self.state_size * self.num_classes

        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        position_offsets = torch.arange(self.state_size, dtype=torch.int64) * self.num_classes
        self.register_buffer("position_offsets", position_offsets, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices):
        if indices.ndim != 2 or indices.size(1) != self.state_size:
            raise ValueError(
                f"expected indices with shape [batch, {self.state_size}], got {tuple(indices.shape)}"
            )
        token_ids = indices + self.position_offsets.unsqueeze(0)
        out = F.embedding_bag(token_ids, self.weight, mode="sum")
        if self.bias is not None:
            out = out + self.bias
        return out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        legacy_weight = self.weight.transpose(0, 1)
        destination[prefix + "weight"] = legacy_weight if keep_vars else legacy_weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            loaded_weight = state_dict[weight_key]
            expected_legacy_shape = (self.out_features, self.in_features)
            expected_native_shape = (self.in_features, self.out_features)
            if loaded_weight.shape == expected_legacy_shape:
                state_dict[weight_key] = loaded_weight.transpose(0, 1)
            elif loaded_weight.shape != expected_native_shape:
                error_msgs.append(
                    f"size mismatch for {weight_key}: expected {expected_legacy_shape} or "
                    f"{expected_native_shape}, got {tuple(loaded_weight.shape)}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.0, num_classes=6):
        super(Pilgrim, self).__init__()
        self.dtype = torch.float32
        self.state_size = state_size
        self.num_classes = num_classes
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.output_dim = int(output_dim)
        self.z_add = 0
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        self.input_layer = LegacyCompatibleEmbeddingBagLinear(
            state_size=state_size,
            num_classes=num_classes,
            out_features=hd1,
        )
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(hd2, dropout_rate) for _ in range(nrd)]
            )
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, self.output_dim)

    def forward(self, z):
        x = self.input_layer(z.long() + self.z_add).to(self.dtype)

        # Input block
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks is not None:
            for block in self.residual_blocks:
                x = block(x)

        # Output
        x = self.output_layer(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    output_dtype = model_attr(model, "dtype", torch.float32)
    input_dtype = model_attr(model, "input_dtype", None)
    output_dim = int(model_attr(model, "output_dim", 1))
    if output_dim == 1:
        outputs = torch.empty(data.size(0), dtype=output_dtype, device=device)
    else:
        outputs = torch.empty((data.size(0), output_dim), dtype=output_dtype, device=device)

    # Process each batch
    with torch.inference_mode():
        for i in range(0, data.size(0), batch_size):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            batch = data[i:i + batch_size]
            if input_dtype is not None and batch.dtype != input_dtype:
                batch = batch.to(dtype=input_dtype)
            batch_output = model(batch)
            if output_dim == 1:
                outputs[i:i + batch_size] = batch_output.view(-1)
            else:
                outputs[i:i + batch_size] = batch_output.view(batch.size(0), output_dim)

    return outputs
