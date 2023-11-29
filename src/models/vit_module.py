from typing import Any
from lightning import LightningModule
import torch
import torch.nn as nn
import numpy as np


def patchify(batch: torch.Tensor, patch_size: tuple = (16, 16)):
    """
    Patchify the batch of images

    Shape:
        batch: (b, h, w, c)
        output: (b, nh, nw, ph, pw, c)
    """
    b, h, w, c = batch.shape # (n, 224, 224, 3)
    ph, pw = patch_size # (16, 16)
    nh, nw = h // ph, w // pw # (14, 14)

    patches = torch.zeros(b, nh*nw, ph*pw*c).to(batch.device) # (n, nh*nw, ph*pw*c) = (n, 196, 768)

    for idx, image in enumerate(batch):
        for i in range(nh):
            for j in range(nw):
                patch = image[i*ph: (i+1)*ph, j*pw: (j+1)*pw, :]
                patches[idx, i*nh + j] = patch.flatten()
    return patches # (n, nh*nw, ph*pw*c) = (n, 196, 768)

def get_mlp(in_features, hidden_units, out_features):
    """
    Returns a MLP head
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result # (s, d)

class ViTModule(LightningModule):
    def __init__(self, learning_rate: float = 1e-4,
                 nhead: int = 4,
                 dim_feedforward: int = 1024,
                 blocks: int = 4,
                 mlp_head_units: list = [1024, 512],
                 n_classes: int = 1,
                 img_size: tuple = (224, 224),
                 patch_size: tuple = (16, 16),
                 n_channels: int = 3,
                 d_model: int = 512) -> None:
        '''
        Args:
            img_size: Size of the image
            patch_size: Size of the patch
            n_channels: Number of image channels
            d_model: The number of features in the transformer encoder
            nhead: The number of heads in the multiheadattention models
            dim_feedforward: The dimension of the feedforward network model in the encoder
            blocks: The number of sub-encoder-layers in the encoder
            mlp_head_units: The hidden units of mlp_head
            n_classes: The number of output classes

        Shape:
            input: (b, c, h, w)
            output: (b, n_classes)
        '''
        super().__init__()
        self.learing_rate = learning_rate
        self.patch_size = patch_size # (16, 16)
        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1] # (14, 14)
        n_tokens = nh * nw # 196
        token_dim = patch_size[0] * patch_size[1] * n_channels # 768
        self.first_linear = nn.Linear(token_dim, d_model) # (768, 512)
        self.cls_token = nn.Parameter(torch.randn(1, d_model)) # (1, 512)
        self.pos_emb = nn.Parameter(get_positional_embeddings(n_tokens, d_model)) # (196, 512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, blocks
        )
        self.mlp = get_mlp(d_model, mlp_head_units, n_classes) # (512, [1024, 512], n_classes)

        self.classifer = nn.Sigmoid() if n_classes == 1 else nn.Softmax()
        # self.criteria = nn.CrossEntropyLoss()

        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            input: (b, c, h, w)
            output: (b, n_classes)
        """
        # batch = self.img2seq(batch) # (b, s, d)
        batch = torch.permute(batch, (0, 2, 3, 1)) # (b, h, w, c) = (b, 224, 224, 3)
        batch = patchify(batch, self.patch_size) # (b, nh*nw, ph*pw*c) = (b, 196, 768)
        b = batch.shape[0]
        batch = self.first_linear(batch) # (b, nh*nw, d_model) = (b, 196, 512)
        cls = self.cls_token.expand([b, -1, -1]) # (b, 1, d_model) = (b, 1, 512)
        emb = batch + self.pos_emb # (b, nh*nw, d_model) = (b, 196, 512)
        batch = torch.cat([cls, emb], axis=1) # (b, nh*nw+1, d_model) = (b, 197, 512)

        batch = self.transformer_encoder(batch) # (b, s, d)
        batch = batch[:, 0, :] # (b, d)
        batch = self.mlp(batch) # (b, n_classes)
        output = self.classifer(batch) # (b, n_classes)
        return output

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy.append(accuracy)
        self.val_loss.append(loss)
        return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_accuracy_epoch', torch.stack(self.train_accuracy).mean())
        self.log('train_loss_epoch', torch.stack(self.train_loss).mean())
        self.train_accuracy = []
        self.train_loss = []

    def on_validation_epoch_end(self) -> None:
        self.log('val_accuracy_epoch', torch.stack(self.val_accuracy).mean())
        self.log('val_loss_epoch', torch.stack(self.val_loss).mean())
        self.val_accuracy = []
        self.val_loss = []

    def on_test_epoch_end(self) -> None:
        self.log('test_accuracy_epoch', torch.stack(self.test_accuracy).mean())
        self.log('test_loss_epoch', torch.stack(self.test_loss).mean())
        self.test_accuracy = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learing_rate)

class ViTPretrainedModule(LightningModule):
    def __init__(self, model, learning_rate: float, source: str = 'pytorch', n_classes: int = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.source = source
        self.learing_rate = learning_rate
        self.model = model
        if n_classes is not None and source == 'pytorch':
            self.model.heads = nn.Linear(self.model.heads.head.in_features, n_classes)
        self.criteria = nn.CrossEntropyLoss()

        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits if self.source == 'huggingface' else self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy.append(accuracy)
        self.val_loss.append(loss)
        return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_accuracy_epoch', torch.stack(self.train_accuracy).mean())
        self.log('train_loss_epoch', torch.stack(self.train_loss).mean())
        self.train_accuracy = []
        self.train_loss = []

    def on_validation_epoch_end(self) -> None:
        self.log('val_accuracy_epoch', torch.stack(self.val_accuracy).mean())
        self.log('val_loss_epoch', torch.stack(self.val_loss).mean())
        self.val_accuracy = []
        self.val_loss = []

    def on_test_epoch_end(self) -> None:
        self.log('test_accuracy_epoch', torch.stack(self.test_accuracy).mean())
        self.log('test_loss_epoch', torch.stack(self.test_loss).mean())
        self.test_accuracy = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learing_rate)
