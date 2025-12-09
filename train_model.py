from utility.face_dataloader import EmoticFaceDataset
from torch.utils.data import DataLoader

dataset = EmoticFaceDataset("data/emotic_faces_128/labels.csv")

loader = DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(loader))
print(images.shape, labels)