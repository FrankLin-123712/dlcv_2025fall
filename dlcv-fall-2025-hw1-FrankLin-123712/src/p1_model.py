import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLPHead2(nn.Module):
    def __init__(
        self,
        in_dim=2048,
        hidden1=1024,
        hidden2=512,
        num_classes=65,
        dropout=0.2,
        use_bn=True,
        act="gelu",   # "relu" or "gelu"
    ):
        super().__init__()
        Act = nn.GELU if act.lower()=="gelu" else nn.ReLU

        layers = [
            nn.Linear(in_dim, hidden1, bias=not use_bn),
            nn.BatchNorm1d(hidden1) if use_bn else nn.Identity(),
            Act(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2, bias=not use_bn),
            nn.BatchNorm1d(hidden2) if use_bn else nn.Identity(),
            Act(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes),
        ]
        self.net = nn.Sequential(*layers)

        # Kaiming init for ReLU/GELU MLPs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, in_dim)
        features = self.net[:-1](x)          # shape [B, hidden2]
        logits = self.net[-1](features)      # shape [B, num_classes]
        return logits, features

def build_resnet50_head(num_classes=65):
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    head = MLPHead2(in_dim=2048, num_classes=num_classes, act="relu") # head
    model = nn.Sequential(backbone, head)
    return model, backbone, head

    
if __name__ == '__main__':
    model, backbone, head = build_resnet50_head()
    print(f"model: ")
    print(f"{model}")
    print(f"model state dict:")
    print(f"{model.state_dict()}")
