import torchvision.models as models
import torch.nn as nn
import torch 

class ConvNext_tiny(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.convnext_tiny(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=768, out_features=out_dim, p=p)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

class ConvNext_base(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.convnext_base(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=1024, out_features=out_dim, p=p)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

    
class ConvNext_large(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.convnext_large(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=1536, out_features=out_dim, p=p)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.resnet34(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=512, out_features=out_dim, p=p)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.resnet50(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=2048, out_features=out_dim, p=p)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        model = models.resnet101(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC = Linear_block(in_features=2048, out_features=out_dim, p=p)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

class VIT_b_16(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        
        super().__init__()

        self.model = models.vit_b_16(weights='DEFAULT')
        self.model.heads.head = Linear_block(in_features=768, out_features=out_dim, p=p)

    def forward(self, x):
        return self.model(x)

class VIT_l_16(nn.Module):
    def __init__(self, out_dim=20, p=0.2):
        super().__init__()

        self.model = models.vit_l_16(weights='DEFAULT')
        self.model.heads.head = Linear_block(in_features=1024, out_features=out_dim, p=p) 

    def forward(self, x):
        return self.model(x)
    
class ConvBN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) :
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
class Linear_block(nn.Module):
    def __init__(self, in_features, out_features, p=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Dropout(p=p)
        )

    def forward(self, x):
        return self.block(x)

class Landmark_regression_model(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 emb_dim=128,
                 outs=28,
                 position_dim=3,
                 positional_encoding=True,
                 p=0.2):
        
        super().__init__()

        self.positional_encoding = positional_encoding
        self.outs = outs

        if backbone == 'resnet34':
            self.backbone = ResNet34(out_dim=emb_dim, p=p) 
        elif backbone == 'resnet50':
            self.backbone = ResNet50(out_dim=emb_dim, p=p)
        elif backbone == 'resnet101':
            self.backbone = ResNet101(out_dim=emb_dim, p=p) 
        elif backbone == 'convnext_tiny':
            self.backbone = ConvNext_tiny(out_dim=emb_dim, p=p)
        elif backbone == 'convnext_base':
            self.backbone = ConvNext_base(out_dim=emb_dim, p=p) 
        elif backbone == 'convnext_large':
            self.backbone = ConvNext_large(out_dim=emb_dim, p=p)
        elif backbone == 'vit_b_16':
            self.backbone = VIT_b_16(out_dim=emb_dim, p=p)  
        elif backbone == 'vit_l_16':
            self.backbone = VIT_l_16(out_dim=emb_dim, p=p)

        print(f'loaded {backbone}')
            
        if positional_encoding:
            self.positional_encoding_emb = Linear_block(in_features=position_dim, out_features=16, p=p)
            self.regression_head = nn.Sequential(
                                        Linear_block(in_features=emb_dim + 16, out_features=emb_dim*2, p=p),
                                        nn.Linear(emb_dim*2, outs)
                                        )
        else:
            self.regression_head = nn.Sequential(
                                        Linear_block(in_features=emb_dim, out_features=emb_dim*2, p=p),
                                        nn.Linear(emb_dim*2, outs)
                                        )

    def forward(self, image, current_position=None):
        batch_size, C, H, W = image.shape
        z_image = self.backbone(image)
        z_image = z_image.view(batch_size, z_image.shape[1]) # [batch_size, emb_dim]
        
        if self.positional_encoding:
            assert current_position is not None, "Please input the current position"
            z_position = self.positional_encoding_emb(current_position)
            z_fusion = torch.cat([z_image, z_position], dim=-1)  # [batch_size, emb_dim + 16]
            output = self.regression_head(z_fusion)
            mean = torch.tanh(output[:, :self.outs//2])  # [batch_size, outs//2]
            variance = nn.functional.softplus(output[:, self.outs//2:])  # [batch_size, outs//2]
            return mean, variance
            
        else:
            return self.regression_head(z_image)
    