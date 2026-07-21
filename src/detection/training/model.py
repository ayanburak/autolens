"""Faster R-CNN nesne tespiti modelini kuran modül."""

import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int) -> nn.Module:
    """Verilen kategori sayısına göre kurulmuş bir Faster R-CNN modeli döndürülür.

    Args:
        num_classes: Arka plan dahil, modelin ayırt edeceği kategori sayısı.

    Returns:
        Eğitime veya ağırlık yüklemeye hazır `nn.Module`.
    """
    #
    # Önceden eğitilmiş, COCO'nun 91 kategorisine göre kurulu bir Faster
    # R-CNN modeli yüklenir; backbone ve RPN kısımları bu hâliyle kullanılır.
    #
    # backbone: görselden özellik haritası çıkaran katmanlar.
    # RPN: bu özellik haritası üzerinde nesne olabilecek bölgelerin önerildiği alt ağ.
    #
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Mevcut sınıflandırma katmanının kaç girdi kabul ettiği öğrenilir;
    # yeni katman da aynı girdi boyutuna göre kurulacaktır.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # num_classes kategoriye göre yeni bir sınıflandırma/kutu tahmin
    # katmanı kurulur; bu katman henüz eğitilmemiştir.
    new_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modelin varsayılan, 91 kategorilik katmanı, yeni kurulan katmanla
    # değiştirilir; backbone ve RPN'e dokunulmaz.
    model.roi_heads.box_predictor = new_predictor

    return model
