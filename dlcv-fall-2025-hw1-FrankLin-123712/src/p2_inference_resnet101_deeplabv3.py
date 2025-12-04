import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101

class SegmentationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        self.model = deeplabv3_resnet101(weights=None)
        self.model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        return self.model(x)['out']

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('_sat.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

def main(test_dir, output_dir, model_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set up model
    num_classes = 7
    model = SegmentationModel(num_classes).to(device)

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    test_dataset = TestDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Class mapping (reverse of what's used in training)
    class_mapping = {0: 3, 1: 6, 2: 5, 3: 2, 4: 1, 5: 7, 6: 0}

    # Inference loop
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Convert predictions back to original format
            for pred, img_name in zip(preds, img_names):
                mask = np.zeros((pred.shape[-2], pred.shape[-1], 3), dtype=np.uint8)
                for class_idx, color_idx in class_mapping.items():
                    mask[pred == class_idx] = [
                        255 if color_idx & 4 else 0,
                        255 if color_idx & 2 else 0,
                        255 if color_idx & 1 else 0
                    ]

                # Save the mask
                output_name = img_name.replace('_sat.jpg', '_mask.png')
                output_path = os.path.join(output_dir, output_name)
                Image.fromarray(mask).save(output_path)

    print(f'Inference completed. Results saved in {output_dir}')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 script.py <test_image_dir> <output_dir> <model_path> <work_dir>")
        sys.exit(1)

    test_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    working_dir = sys.argv[4]

    # Set working directory (modify this path as needed)
    os.chdir(working_dir)
    print(f"Working directory set to: {os.getcwd()}")

    main(test_dir, output_dir, model_path)