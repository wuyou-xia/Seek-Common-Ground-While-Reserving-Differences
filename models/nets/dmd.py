"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models
from math import sqrt
# from .multiway_transformer import Attention

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Modal_Select(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Modal_Select, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        
        # 输入特征的维度
        self.in_dim = in_dim
        
        # 定义查询、键和值权重矩阵
        self.query_weight = nn.Linear(in_dim, in_dim)
        self.key_weight = nn.Linear(in_dim, in_dim)
        self.value_weight = nn.Linear(in_dim, in_dim)
        
        # 定义一个可学习的缩放参数
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # self.scale = sqrt(in_dim)
        
    def forward(self, x):
        # 计算查询、键和值
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, value)
        
        return output

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = models.resnet18(pretrained=True)
        self.visual_model = nn.Sequential(*list(self.visual_model.children())[:-1])
        self.v_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.t_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.m_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.attention = Attention(in_dim=512)
        # self.modal_select_layer = nn.Linear(in_features=512*3, out_features=3)
        self.modal_select_layer = Modal_Select(input_size=512*3,hidden_size=512, num_classes=3)


        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(768, 512, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.1 Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(512*2, 512, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(512*2, 512, kernel_size=1, padding=0, bias=False)



    def forward(self, image, text):
        # if self.use_bert:
        text = self.text_model(**text).pooler_output
        image = self.visual_model(image).squeeze()
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        # x_v = image.transpose(1, 2)

        text = self.proj_l(text.unsqueeze(2))
        image = self.proj_v(image.unsqueeze(2))
        # image = image.unsqueeze(2)
        # proj_x_v = self.proj_v(x_v)

        # Modality-specific feature
        s_l = self.encoder_s_l(text).squeeze()
        s_v = self.encoder_s_v(image).squeeze()
        pre_t = self.t_classifier(s_l)
        pre_v = self.v_classifier(s_v)

        # Modality-common feature
        c_l = self.encoder_c(text).squeeze()
        c_v = self.encoder_c(image).squeeze()
        c_m = c_l+c_v
        c_list = [c_l, c_v]
        pre_m = self.m_classifier(c_m)
        pre_m_in_v = self.v_classifier(c_m)
        pre_m_in_t = self.t_classifier(c_m)
        pre_v_in_m = self.m_classifier(s_v)
        pre_t_in_m = self.m_classifier(s_l)

        c_l_sim = c_l
        c_v_sim = c_v

        # decoder 
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1).unsqueeze(2))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1).unsqueeze(2))


        s_l_r = self.encoder_s_l(recon_l).squeeze()
        s_v_r = self.encoder_s_v(recon_v).squeeze()

        select_modal = self.modal_select_layer(torch.cat((s_l,s_v,c_m),dim=1))
        modal_index = torch.argmax(torch.softmax(select_modal,dim=1),dim=1)

        # attention
        s_l_att = s_l.unsqueeze(1)
        s_v_att = s_v.unsqueeze(1)
        c_m_att = c_m.unsqueeze(1)
        att_tensor = torch.cat((s_l_att, s_v_att, c_m_att), dim=1)
        att_m = self.attention(att_tensor)
        select_m = att_m[0][modal_index[0]].unsqueeze(0)
        for i in range(1,len(modal_index)):
            select_m = torch.concat((select_m,att_m[i][modal_index[i]].unsqueeze(0)),dim=0)    
        # select_m = select_m.unsqueeze(1)
        # modal_index = modal_index.unsqueeze(0)
        # att_m = torch.gather(att_m,1,modal_index)
        # [:,modal_index,:]
        pre_m_att = self.m_classifier(select_m)

        res = {
            # 'origin_l': proj_x_l,
            # 'origin_v': proj_x_v,
            'origin_l': text,
            'origin_v': image,
            's_l': s_l,
            's_v': s_v,

            'c_l': c_l,
            'c_v': c_v,

            's_l_r': s_l_r,
            's_v_r': s_v_r,

            'recon_l': recon_l,
            'recon_v': recon_v,

            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,

            'att_m': att_m,
            # 'modal_index': modal_index,

            'pre_t': pre_t,
            'pre_v': pre_v,
            'pre_m': pre_m,
            'pre_m_att': pre_m_att,

            'pre_m_in_t': pre_m_in_t,
            'pre_m_in_v': pre_m_in_v,
            'pre_v_in_m': pre_v_in_m,
            'pre_t_in_m': pre_t_in_m,

            'modal_index': modal_index
        }
        return res