from torch import nn
import torch

from image_encoder_tq import patch_norm
    
class Embeddings(nn.Module):
    def __init__(self, width: int, input_resolution: int, patch_size: int, in_channels: int, hidden_dropout_prob: float):
        super().__init__()
        self.scale = width ** -0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(self.scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = patch_norm('bn_1d', width)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor):
        #x shape = [batch_size, num_channels, height, width]
        x = self.conv1(x)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #x shape = [batch_size, num_patches, embed_dim]
        x = x.permute(0,2,1)
        x = self.ln_pre(x)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #x shape = [batch_size, num_patches, embed_dim]

        x = x + self.positional_embedding.to(x.dtype)
        x= self.dropout(x)
        return x
    
class PretrainModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
            