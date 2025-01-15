import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

class Dianxian(Dataset):
    def __init__(self, feature1, adj1, feature2, adj2, labels):
        super(Dianxian, self).__init__()
        self.feature1 = feature1
        self.adj1 = adj1
        self.feature2 = feature2
        self.adj2 = adj2
        self.labels = labels

    def __getitem__(self, item):
        feature1 = self.feature1[item]
        adj1 = self.adj1[item]
        feature2 = self.feature2[item]
        adj2 = self.adj2[item]
        labels = self.labels[item]
        return feature1, adj1, feature2, adj2, labels

    def __len__(self):
        return self.feature1.shape[0]


def load_data(fmri_path, dti_path, num_window, drop_prob1=0.075, drop_prob2=0.125):
    # 加载 fMRI 数据
    fmri_data = loadmat(fmri_path)
    feature = fmri_data[list(fmri_data.keys())[3]][0:217]
    net_all = [np.corrcoef(feature[i]) for i in range(feature.shape[0])]
    fdata = np.array(net_all)
    labels = fmri_data[list(fmri_data.keys())[4]][0][0:217]

    # 加载 DTI 数据
    dti_data = loadmat(dti_path)
    ddata = dti_data[list(dti_data.keys())[3]].transpose(2, 0, 1)[0:217]

    # 数据增强
    def drop_value(x, drop_prob):
        drop_mask = np.random.uniform(0, 1, size=x.shape) < drop_prob
        x[drop_mask] = 0
        return x

    x1 = drop_value(fdata, drop_prob1)
    adj1 = drop_value(ddata, drop_prob1)
    x2 = drop_value(fdata, drop_prob2)
    adj2 = drop_value(ddata, drop_prob2)

    return Dianxian(x1, adj1, x2, adj2, labels)