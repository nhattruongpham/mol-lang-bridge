import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init as init

class GFL(nn.Module):
    def __init__(self, support_f1_input_dim, support_f2_input_dim, main_f3_input_dim, gated_dim):
        super(GFL, self).__init__()
        
        self.WF1 = nn.Parameter(torch.Tensor(support_f1_input_dim, gated_dim))
        self.WF2 = nn.Parameter(torch.Tensor(support_f2_input_dim, gated_dim))
        
        init.xavier_uniform_(self.WF1)
        init.xavier_uniform_(self.WF2)
        
        dim_size_f = support_f1_input_dim + support_f2_input_dim
        
        self.WF = nn.Parameter(torch.Tensor(dim_size_f, gated_dim))
        
        init.xavier_normal_(self.WF)
        
    def forward(self, f1, f2, f3):
        h_f1 = torch.tanh(torch.matmul(f1, self.WF1))
        h_f2 = torch.tanh(torch.matmul(f2, self.WF2))
        z_f = torch.softmax(torch.matmul(torch.cat([f1,f2], dim=1), self.WF), dim=1)
        h_f = z_f * h_f1 + (1-z_f)*h_f2
        return h_f
    
class Fusion(nn.Module):
    def __init__(self, 
                 input_dim_f1,
                 input_dim_f2,
                 input_dim_f3,
                 gated_dim):
        super(Fusion, self).__init__()
        
        self.fc_f1 = nn.Linear(input_dim_f1, input_dim_f1)
        self.fc_f2 = nn.Linear(input_dim_f2, input_dim_f2)
        self.fc_f3 = nn.Linear(input_dim_f3, input_dim_f3)
        
        init.xavier_uniform_(self.fc_f1.weight)
        init.xavier_uniform_(self.fc_f2.weight)
        init.xavier_uniform_(self.fc_f3.weight)
        
        self.gfl_f1f2 = GFL(input_dim_f1, input_dim_f2, gated_dim)
        self.gfl_f2f3 = GFL(input_dim_f2, input_dim_f3, gated_dim)
        self.gfl_f3f1 = GFL(input_dim_f3, input_dim_f1, gated_dim)
        
        self.final_fc = nn.Linear(gated_dim * 3, gated_dim)
        init.xavier_uniform_(self.final_fc.weight)
    
    
    def forward(self, input_f1, input_f2, input_f3):
        fc_f1 = self.fc_f1(input_f1)
        fc_f2 = self.fc_f2(input_f2)
        fc_f3 = self.fc_f3(input_f3)
        
        h_f1f2 = self.gfl_f1f2(fc_f1, fc_f2)
        h_f2f3 = self.gfl_f2f3(fc_f2, fc_f3)
        h_f3f1 = self.gfl_f3f1(fc_f3, fc_f1)
        
        h_combined = torch.cat([h_f1f2, h_f2f3, h_f3f1], dim=1)
        h_final = self.final_fc(F.relu(h_combined))
        
        return h_final
    
    # print(sum(p.numel() for p in model.parameters()))