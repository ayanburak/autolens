"""
COCO formatındaki nesne tespiti verisini PyTorch `Dataset` olarak yükleyen modül.
"""
import json, os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CocoDetectionDataset(Dataset):
    """COCO formatındaki bbox etiketlerini okuyup `Dataset` arayüzünü dolduran sınıf."""

    def __init__(self, images_dir: str, annotations_path: str):
        """
        Args:
            images_dir: Görsel dosyalarının bulunduğu klasörün yolu.
            annotations_path: `_annotations.coco.json` dosyasının yolu.
        """
        self.images_dir = images_dir
        self.annotations_path = annotations_path

        # annotations_path'teki json dosyası açılıp Python sözlüğüne çevrilir.
        with open(annotations_path, 'r', encoding='utf-8') as dosya:
            data = json.load(dosya)
            # "images" listesindeki her elemanda bir görsele ait id, dosya
            # adı ve boyut bilgisi bulunur. self.images'e kaydedilir;
            # böylece __len__ ve __getitem__ tarafından da erişilebilir.
            self.images = data["images"]

        # "annotations" listesi düz bir liste olarak gelir; her elemanında
        # hangi görsele ait olduğunu belirten bir image_id alanı bulunur
        # ve bir görselde birden fazla plaka olabildiğinden aynı image_id
        # birden fazla kez tekrarlanabilir. Aşağıdaki döngüyle bu liste
        # image_id'ye göre gruplanıp bir sözlüğe çevrilir; böylece bir
        # görselin bbox'ları, tüm liste baştan sona taranmadan, __getitem__
        # tarafından doğrudan image_id ile bulunabilir.
        self.annotations_by_image = {}
        for annotation in data["annotations"]:
            # Bu image_id sözlükte ilk kez görülüyorsa, önce boş bir liste açılır.
            if annotation["image_id"] not in self.annotations_by_image:
                self.annotations_by_image[annotation["image_id"]] = []

            # Liste artık kesin var, bu annotation o listeye eklenir.
            self.annotations_by_image[annotation["image_id"]].append(annotation)


    def __len__(self) -> int:
        """Veri setindeki toplam görsel sayısı döndürülür."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Belirtilen indeksteki görsel ve etiketleri döndürülür.

        Args:
            index: Okunacak örneğin sıradaki konumu.

        Returns:
            Piksel matrisi ve bu görsele ait bbox/kategori etiketlerini içeren ikili.
        """
        # self.images listesinde index'inci sıradaki görselin id, dosya
        # adı ve boyut bilgisini içeren sözlüğü alınır.
        index_dict = self.images[index]

        # Görselin klasördeki tam dosya yolu, images_dir ile file_name
        # birleştirilerek oluşturulur.
        file_path = os.path.join(self.images_dir, index_dict["file_name"])

        # Dosya diskten açılır; convert("RGB") ile renk modu her zaman
        # rgb üç kanala sabitlenir.
        image = Image.open(file_path).convert("RGB")

        # PIL görsel nesnesi, eğitimde kullanılacak, yükseklik-genişlik-kanal
        # sırasıyla sayısal piksel matrisine çevrilir.
        pixel_matrix = np.array(image)

        # Bu görselin annotation'ları, index_dict'in id alanı kullanılarak
        # __init__'te kurulan sözlükten bulunur. Bu id, döngüdeki index
        # değeriyle aynı olabilir ama kavramsal olarak ondan bağımsız bir alandır.
        annotations = self.annotations_by_image[index_dict["id"]]
        boxes = []
        labels = []

        for annotation in annotations:
            # COCO'nun bbox formatı [x, y, genişlik, yükseklik] şeklindedir;
            # x, y sol-üst köşenin piksel koordinatıdır.
            x, y, w, h = annotation["bbox"]

            # Faster R-CNN tarafından sol-üst ve sağ-alt köşe olan
            # [x1, y1, x2, y2] formatı beklendiği için burada bu dönüşüm yapılır.
            boxes.append([x, y, x + w, y + h])

            # Bu bbox'un hangi kategoriye ait olduğu ayrı bir listede,
            # boxes ile aynı sırada tutulur.
            labels.append(annotation["category_id"])

        # Piksel matrisi ve bbox/kategori bilgisini içeren sözlük,
        # bir tuple olarak döndürülür.
        return pixel_matrix, {"boxes": boxes, "labels": labels}
