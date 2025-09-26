"""
SupConLoss (Supervised Contrastive Loss) fonksiyonu,
PyTorch için orijinal makaleye dayanarak yeniden yazılmıştır.
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning için Kayıp Fonksiyonu.

    `features` ve `labels`'tan oluşan bir mini-batch üzerinde
    kullanılmak üzere tasarlanmıştır.

    Args:
        temperature: Skaler değer, benzerlik skorlarını yumuşatmak için kullanılır.
        contrast_mode: 'all' veya 'one_vs_one'. 'all' modu tüm pozitif çiftleri kullanır.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Kayıp fonksiyonunun ileri yayılımı (forward pass).

        Args:
            features: Şekli `[N, D]` olan tensör, burada N batch boyutu, D özellik boyutu.
            labels: Şekli `[N]` olan tensör, her örneğin sınıf etiketlerini içerir.
            mask: Şekli `[N, N]` olan ikili (binary) maske, hangi çiftlerin
                  pozitif olduğunu gösterir. Genellikle `labels`'dan türetilir.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features`ın en az 2 boyutu olmalıdır: [N, D]')
        if features.shape[1] < 1:
            raise ValueError('Özellik boyutu (D) 1den büyük olmalıdır.')

        if labels is not None and mask is not None:
            raise ValueError('Hem `labels` hem de `mask` aynı anda verilemez.')
        if labels is None and mask is None:
            raise ValueError('Ya `labels` ya da `mask` sağlanmalıdır.')

        if mask is None:
            # Maske oluştur: aynı etiketli örnekler pozitif çifttir
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != features.shape[0]:
                raise ValueError('Etiket boyutu özellik boyutu ile eşleşmiyor.')
            
            mask = torch.eq(labels, labels.T).float().to(device)

        # Kosinüs benzerliği hesapla
        anchor_feature = features
        contrast_feature = features

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # Kendini dışla (anchor ile anchor'ın benzerliği hariç)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        ).to(device)
        mask = mask * logits_mask

        # Log-likeliood'u hesapla
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

        # Kaybı hesapla: her pozitif çift için ortalama log-prob
        loss = -(self.temperature / self.base_temperature) * torch.sum(mask * log_prob) / (mask.sum(1).clamp(min=1e-8)).sum()

        return loss
