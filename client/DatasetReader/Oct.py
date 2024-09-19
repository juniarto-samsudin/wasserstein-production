from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.classes = ('NORMAL', 'CMV', 'DNE', 'DRUSEN')
        if transform is None:
            transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            self.transform = transform
        # Create a mapping from labels to indices
        for label in sorted(os.listdir(root_dir)):
            label_idx = len(self.label_map)
            self.label_map[label] = label_idx
            for img_name in os.listdir(os.path.join(root_dir, label)):
                self.image_paths.append(os.path.join(root_dir, label, img_name))
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label