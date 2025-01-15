from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import Parameter, Module, init
import torch.nn.functional as F


class GATLayer(Module): 
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = 0.5
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W = Parameter(torch.empty(in_features, out_features))  # W是权重矩阵
        init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))  # a是一个向量，用来将高维特征映射到一个实数上
        init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):  # node:1.fMRI , edge:DTI
        batch_size = adj.size(0)
        Wh = torch.matmul(h, self.W) 
        e = self.prepare_attentional_mechanism_input(Wh, batch_size)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # 归一化，化成概率，便是重要性
        attention = F.dropout(attention, self.dropout)
        embedding = torch.matmul(attention, Wh)  # GAT输出的对于每个顶点的新特征（融合了邻域信息）
        return embedding

    def prepare_attentional_mechanism_input(self, Wh, batch_size):  # 计算注意力系数
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = Wh2.view(batch_size, 1, -1)
        e = Wh1 + Wh2
        return self.leakyrelu(e)
    
class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

    def forward(self, x):  # 输入的是feature和adj邻接矩阵
        return self.layers(x) 

class Attention(nn.Module):
    def __init__(self, in_size=128, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
    
class IntraLoss(nn.Module):
    def __init__(self):
        super(IntraLoss, self).__init__()
        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, z1, z2): 
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()                     
        return ret     
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def semi_loss(self, z1, z2):
        tau = 0.4
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
class InterLoss(nn.Module):
    def __init__(self):
        super(InterLoss, self).__init__()
        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, z1, z2): 
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()                     
        return ret     
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def semi_loss(self, z1, z2):
        tau = 0.4
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))    

class MyNetwork(nn.Module):
    def __init__(self, nfeat, nhid, num_window):
        super(MyNetwork, self).__init__()
        # 以下的编码器和解码器是共享参数的
        self.MC_GAT = nn.ModuleList([GATLayer(nfeat, nhid) for i in range(num_window )])
        self.GAT = GATLayer(nfeat, nhid)
        self.Projection = nn.ModuleList([ProjectionHead() for i in range(num_window )])
        self.attention = Attention()
        
        self.intra_loss = IntraLoss()
        self.inter_loss = InterLoss()
        # LSTM模块
        self.LSTM = nn.LSTM(input_size=128, hidden_size = 64, num_layers=2, batch_first = True)
        # MLP分类器，加了很多bn和dropout
        self.f1 = nn.Flatten()
        self.bn2 = torch.nn.BatchNorm1d(128, eps=1e-05)
        self.bn3 = torch.nn.BatchNorm1d(64, eps=1e-05)
        self.bn4 = torch.nn.BatchNorm1d(32, eps=1e-05)
        self.l1 = nn.Linear(192, 128)  # 24300
        self.d1 = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(128, 64)
        self.d2 = nn.Dropout(p=0.5)
        self.l3 = nn.Linear(64, 32)
        self.d3 = nn.Dropout(p=0.5)
        self.l4 = nn.Linear(32, 2)
        # self.logs = nn.LogSoftmax(dim=1)

    def forward(self, x1, adj1, x2, adj2):  # 输入的是feature和adj邻接矩阵
        batch_size = x1.size(0)  # batchsize
        time_length = x1.size(1) # window
        ######### 多通道GAT开始提取特征
        graph1 = []
        for i, gat_layer in enumerate(self.MC_GAT):
            features = x1[:, i, :, :]
            adj_new = adj1
            temp = gat_layer(features, adj_new)
            out =  F.relu(gat_layer(temp , adj_new))
            graph1.append(out)
        graph1 = torch.stack(graph1, dim=1)  #(16,3,90,90)

        graph2 = []
        for i, gat_layer in enumerate(self.MC_GAT):
            features = x2[:, i, :, :]
            adj_new = adj2
            temp = gat_layer(features, adj_new)
            out =  F.relu(gat_layer(temp , adj_new))
            graph2.append(out)
        graph2 = torch.stack(graph2, dim=1)  #(16,3,90,90)

        ######### 多通道Projection开始降维
        list1 = []
        a1 = graph1.view(batch_size, time_length, -1)
        for i, pro_layer in enumerate(self.Projection):
            temp = a1[:, i, :]
            temp = pro_layer(temp )
            list1.append(temp)
        list1= torch.stack(list1, dim=1) #(16,3,128)       

        list2 = []
        a2 = graph2.view(batch_size, time_length, -1)
        for i, pro_layer in enumerate(self.Projection):
            temp = a2[:, i, :]
            temp = pro_layer(temp )
            list2.append(temp)
        list2= torch.stack(list2, dim=1) #(16,3,128) 
        
        ######### 注意力机制
        stack = torch.stack([list1, list2], dim=1)
        out_spatio, att = self.attention(stack)#(16,3,128)

        #######intra_loss
        embedding1 = list1.view((-1,128)) #*(16,128)
        embedding2 = list2.view((-1,128)) 
        intra_loss = self.intra_loss(embedding1, embedding2)
        ########inter_loss
        inter_loss = 0.0
        for i in range(time_length):
            for j in range(i+1, time_length):
                win_i = out_spatio[:, i, :]
                win_j = out_spatio[:, j, :]
                temp = self.inter_loss(win_i, win_j)
                inter_loss  = inter_loss  + temp

        ######## LSTM开始提取时间特征
        h0 = torch.zeros(2, batch_size, 64).to(DEVICE)
        c0 = torch.zeros(2, batch_size, 64).to(DEVICE)
        out_spatio_temporal , (hn, cn) = self.LSTM(out_spatio, (h0, c0))  #(16,3,64)

        ######## 分类器
        outs = self.f1(out_spatio_temporal) #(16,192)
        out = self.d1(F.relu(self.bn2(self.l1(outs))))
        # out = F.relu(self.bn2(out))
        out = self.d2(F.relu(self.bn3(self.l2(out))))
        # out = F.relu(self.bn3(out))
        out = self.d3(F.relu(self.bn4(self.l3(out))))
        # tsne = out.detach().cpu().numpy()
        out = self.l4(out)
        out = F.log_softmax(out, dim=1)
        return out, intra_loss, inter_loss