import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(AttentionModule, self).__init__()
        
        # 输入特征的维度
        self.in_dim = in_dim
        
        # 定义查询、键和值权重矩阵
        self.q_t = nn.Linear(in_dim, in_dim)
        self.k_t = nn.Linear(in_dim, in_dim)
        self.v_t = nn.Linear(in_dim, in_dim)

        self.q_v = nn.Linear(in_dim, in_dim)
        self.k_v = nn.Linear(in_dim, in_dim)
        self.v_v = nn.Linear(in_dim, in_dim)

        self.q_m = nn.Linear(in_dim, in_dim)
        self.k_m = nn.Linear(in_dim, in_dim)
        self.v_m = nn.Linear(in_dim, in_dim)

        self.modal_dict = {0:'t', 1:'v', 2:'m'}
        
        # 定义一个可学习的缩放参数
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        
    def forward(self, t, v, m, modal):
        # 计算查询、键和值
        q_t= self.q_t(t)
        k_t = self.k_t(t)
        v_t = self.v_t(t)

        q_v= self.q_t(v)
        k_v = self.k_t(v)
        v_v = self.v_t(v)

        q_m= self.q_t(m)
        k_m = self.k_t(m)
        v_m = self.v_t(m)

        if self.modal_dict[modal] == 't':
            q = q_t
        elif self.modal_dict[modal] == 'v':
            q = q_v
        else:
            q = q_m

        att_t = (q @ k_t.T / self.scale)
        att_v = (q @ k_v.T / self.scale)
        att_m = (q @ k_m.T / self.scale)
        att_t_weight = torch.softmax(att_t, dim=-1)
        att_v_weight = torch.softmax(att_v, dim=-1)
        att_m_weight = torch.softmax(att_m, dim=-1)

        att_tensor = torch.cat((att_t.unsqueeze(1), att_v.unsqueeze(1), att_m.unsqueeze(1)), dim=1)
        attn_weights = torch.softmax(att_tensor, dim=1)

        # value =  torch.cat((v_t.unsqueeze(1), v_t.unsqueeze(1), v_t.unsqueeze(1)), dim=1)
        # 计算注意力分数
        # attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # 对注意力分数进行softmax操作
        # attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 使用注意力权重对值进行加权求和
        output = torch.matmul(attn_weights, v_m)
        output = 0
        
        return output

if __name__ == "__main__":
    # 创建一个输入张量
    t = torch.rand((32,1, 512)) 
    v = torch.rand((32,1, 512)) 
    m = torch.rand((32, 1,512)) 
    
    # 创建一个注意力模块
    attn_module = AttentionModule(512)
    
    # 使用注意力模块进行特征融合
    output = attn_module(t,v,m,0)
    
    # print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
