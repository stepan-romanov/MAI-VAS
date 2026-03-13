import torch
import torch.nn as nn
import torchvision.models as models
    
def returnResNet(pretrain):
    """Function to return a ResNet50 model with the final layer removed. """
    if pretrain:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet50(weights = weights)
    model.fc = nn.Identity()
    
    return model

def returnViT(pretrain):
    """Function to return a ViT model with the final layer removed. NOT USED."""
    if pretrain:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.vit_b_16(weights = weights)
    model.fc = nn.Identity()
    
    return model

class MAI_VAS_Model(nn.Module):
    """MAI-VAS model class."""
    
    FF_CONFIGS = {
            'resnet50': {
                'feature_dim': 2048,
                'hidden_dim': 1024,
                'output_dim': 512
            },
            'vit': {
                'feature_dim': 768,
                'hidden_dim': 384,
                'output_dim': 192
            }
        }
    
    def __init__(self, pretrain, dropout = 0.5, model = 'resnet50'):
        """
        Initialize the MAI_VAS_Model.

        Args:
            pretrain (bool): Whether to use pretrained weights for the feature extractor.
            dropout (float, optional): Dropout rate for the regressor, by default 0.5.
            model (str, optional): Which feature extractor to use, 'vit' or 'resnet50', by default 'resnet50'.
        """
        super(MAI_VAS_Model, self).__init__()
        
        self.model = model
        self.dropout = dropout

        if self.model == 'resnet50':
            self.extractor = returnResNet(pretrain)
        elif self.model == 'vit':
            self.extractor = returnViT(pretrain)
        else:
            raise ValueError("Model not supported. Choose either 'resnet50' or 'vit'.")
        
        self.feature_dim = self.FF_CONFIGS[self.model]['feature_dim']
        self.hidden_dim  = self.FF_CONFIGS[self.model]['hidden_dim']
        self.output_dim  = self.FF_CONFIGS[self.model]['output_dim']
                  
        # Standard regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),  
            nn.Linear(self.output_dim, 1)
            )
        
    def forward(self, x):
        x = self.extractor(x)
        x = self.regressor(x)
        return x

        