import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer

class Normalize(nn.Module):
    
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class FusionModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='bert-base-uncased'):
        super(FusionModel, self).__init__()

        # Load pre-trained models
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Remove classification heads
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Fusion layer
        resnet_features_dim = 2048  # Output dimension of ResNet-50
        bert_features_dim = self.bert.config.hidden_size
        self.fusion_layer = nn.Linear(resnet_features_dim + bert_features_dim, num_classes)
        # self.fusion_layer = nn.Linear(self.resnet.fc.in_features + self.bert.config.hidden_size, num_classes)
        self.l2norm = Normalize(2)

    def forward(self, image, text):
        # Image feature extraction using ResNet
        image_features = self.resnet(image)
        image_features = image_features.mean(dim=(2, 3))  # Global average pooling
        
        # Text feature extraction using BERT
        text_features = self.bert(**text).pooler_output
        
        # Concatenate image and text features
        fusion_features = torch.cat((image_features, text_features), dim=1)
        # feat = self.fc1(fusion_features)
        # feat = self.relu_mlp(feat)       
        # feat = self.fc2(feat)
        feature = self.l2norm(fusion_features)        
        # Classification using fusion features
        output = self.fusion_layer(fusion_features)
        return output, feature

# # Example usage
# num_classes = 10  # Change this to the number of classes in your classification task
# model = FusionModel(num_classes)

# # Generate some example inputs (you need to replace these with your actual inputs)
# image_input = torch.randn(1, 3, 224, 224)  # Example image input (batch size, channels, height, width)
# text_input = "This is an example sentence."  # Example text input

# # Forward pass through the model
# output = model(image_input, text_input)
# print(output)
