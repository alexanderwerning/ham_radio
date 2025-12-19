import scipy
import torch
import torchaudio

from ham_radio.system.model import SADModel


class ResampleModel(torch.nn.Module):
    def __init__(self, base_model: SADModel, orig_sr=8000, target_sr=16000):
        super().__init__()
        self.base_model = base_model
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=target_sr,
            new_freq=orig_sr,
        )

    def forward(self, x):
        # x.shape: (batch, sequence_length)
        # Resample from target_sr to orig_sr
        x_resampled = self.resampler(x)
        scores = self.base_model(x_resampled)
        score = self.base_model.maybe_pool(scores)

        # output shape: (batch, 100Hz sequence of scores)
        return torch.tile(score, (1, self.target_sr // self.orig_sr))
    
    def vad(self, x):
        vad = self.forward(x).detach().cpu().numpy() > 0.5
        # 100Hz -> 100 = 1s
        vad = scipy.ndimage.maximum_filter1d(vad, 100)  # close small gaps
        vad = scipy.ndimage.minimum_filter1d(vad, 200)  # remove short activations
        vad = scipy.ndimage.maximum_filter1d(vad, 200)  # add collar around speech
        return vad