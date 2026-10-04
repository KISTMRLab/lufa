import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, Wav2Vec2Config, Wav2Vec2Model


class RetrievalModel(nn.Module):
    def __init__(self, audio, text, latent=128):
        super().__init__()
        self.audio = audio
        self.text = text
        self.audio_projection = nn.Linear(audio.config.hidden_size, latent)
        self.text_projection = nn.Linear(text.config.hidden_size, latent)
        self.audio_reconstruction = nn.Sequential(nn.Linear(latent, 256), nn.GELU(), nn.Linear(256, 90 * 9))
        self.text_reconstruction = nn.Sequential(nn.Linear(latent, 256), nn.GELU(), nn.Linear(256, 90 * 9))

    @classmethod
    def initialize(cls, audio_path, text_path, from_scratch, vocab_size):
        if from_scratch:
            audio = Wav2Vec2Model(Wav2Vec2Config(hidden_size=256, num_hidden_layers=4,
                num_attention_heads=4, intermediate_size=1024, conv_dim=(128,) * 7, mask_time_prob=0.))
            text = BertModel(BertConfig(vocab_size=vocab_size, hidden_size=256,
                num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024))
        else:
            audio = Wav2Vec2Model.from_pretrained(audio_path, local_files_only=True)
            text = BertModel.from_pretrained(text_path, local_files_only=True)
        return cls(audio, text)

    def encode_audio(self, samples, mask):
        h = self.audio(samples, attention_mask=mask).last_hidden_state
        lengths = mask.sum(-1)
        for kernel, stride in zip(self.audio.config.conv_kernel, self.audio.config.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        valid = torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]
        pooled = (h * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.audio_projection(pooled), dim=-1)

    def encode_text(self, tokens):
        h = self.text(**tokens).last_hidden_state
        mask = tokens["attention_mask"][:, :, None]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(self.text_projection(pooled), dim=-1)

    def objective(self, samples, mask, tokens, motion, temperature=.07):
        if len(samples) < 2:
            raise ValueError("Contrastive training requires at least two paired clips per batch")
        a, t = self.encode_audio(samples, mask), self.encode_text(tokens)
        target = torch.arange(len(a), device=a.device)
        scores = a @ t.T / temperature
        contrastive = .5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target))
        audio_motion = torch.sigmoid(self.audio_reconstruction(a)).reshape(-1, 90, 9)
        text_motion = torch.sigmoid(self.text_reconstruction(t)).reshape(-1, 90, 9)
        reconstruction = .5 * (F.l1_loss(audio_motion, motion) + F.l1_loss(text_motion, motion))
        return reconstruction + contrastive, reconstruction, contrastive
