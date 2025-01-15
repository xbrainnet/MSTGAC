from scipy.io import loadmat
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter, Module, init
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

seed = 7
torch.manual_seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据
dataset = load_data("path_to_data")
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)



test_acc = []
test_auc = []
test_spe = []
test_sen = []
test_pre = []
test_recall = []
test_f1 = []
label_ten = []
pro_ten = []

kk = 10
# 超参数
bs_train, bs_test = 16, 10
lr = 1e-3
epochs = 300

for train_idx, test_idx in KF.split(dataset):
    train_subsampler = SubsetRandomSampler(train_idx)
    test_sunsampler = SubsetRandomSampler(test_idx)
    datasets_train = DataLoader(dataset, batch_size=bs_train, shuffle=False, sampler=train_subsampler, drop_last=True)
    datasets_test = DataLoader(dataset, batch_size=bs_test, shuffle=False, sampler=test_sunsampler, drop_last=True)
    epoch = 300
    losses = []  # 记录训练误差，用于作图分析
    acces = []
    eval_losses = []
    eval_acces = []
    patiences = 60
    min_acc = 0
    model = MyNetwork(90, 90, num_window)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for num in range(epoch):
        train_loss = 0
        train_acc = 0
        model.train()
        for feature1, adj1, feature2, adj2 ,label in datasets_train:
            feature1, adj1, feature2, adj2 ,label= feature1.to(DEVICE), adj1.to(DEVICE), feature2.to(DEVICE), adj2.to(DEVICE) ,label.to(DEVICE)
            # 前向传播
            feature1 = feature1.float()
            feature2 = feature2.float()
            adj1 = adj1.float()
            adj2 = adj2.float()
            label = label.long()
            out, loss_intra, loss_inter= model(feature1, adj1, feature2, adj2)
            loss_class = F.nll_loss(out, label)
            loss = loss_class + lambda1*loss_intra  + lambda2*loss_inter
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / feature1.shape[0]
            train_acc += acc
        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        # 测试集测试
        eval_loss, eval_acc, eval_acc_epoch, specificity, sensitivity, f1, my_auc,  precision, recall, labels_all, pro_all = test(
            model, datasets_test)
        print(
            'i:{}, epoch:{}, TrainLoss:{:.6f}, TrainAcc:{:.6f}, EvalLoss:{:.6f}, EvalAcc:{:.6f},precision:{'
            ':.6f}, recall:{:.6f}, f1:{:.6f}, my_auc:{:.6f}'
            .format(i, num, train_loss / len(datasets_train), train_acc / len(datasets_train), eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))
        if eval_acc_epoch > min_acc:
            # torch.save(model.state_dict(), '../result2/latest' + str(i) + '.pth')
            print("Model saved at epoch{}".format(num))
            min_acc = eval_acc_epoch
            spe_gd = specificity
            sen_gd = sensitivity
            f1_gd = f1
            auc_gd = my_auc
            pre_gd = precision
            recall_gd = recall
            labels_all_gd = labels_all
            pro_all_gd = pro_all
            patience = 0
        else:
            patience += 1
        if patience > patiences:
            break
        eval_losses.append(eval_loss / len(datasets_test))
        eval_acces.append(eval_acc / len(datasets_test))

    test_acc.append(min_acc)
    test_spe.append(spe_gd)
    test_sen.append(sen_gd)
    test_f1.append(f1_gd)
    test_auc.append(auc_gd)
    test_pre.append(pre_gd)
    test_recall.append(recall_gd)
    label_ten.extend(labels_all_gd)
    pro_ten.extend(pro_all_gd)
    i = i + 1
acc_std = np.std(test_acc)
spe_std = np.std(test_spe)
sen_std = np.std(test_sen)
f1_std = np.std(test_f1)
auc_std = np.std(test_auc)
pre_std = np.std(test_pre)
recall_std = np.std(test_recall)
avg_acc = sum(test_acc) / kk
avg_spe = sum(test_spe) / kk
avg_sen = sum(test_sen) / kk
avg_f1 = sum(test_f1) / kk
avg_auc = sum(test_auc) / kk
avg_pre = sum(test_pre) / kk
avg_recall = sum(test_recall) / kk
print('bs_train', bs_train, 'bs_test', bs_test, 'acc', avg_acc)
print('bs_train', bs_train, 'bs_test', bs_test, 'spe', avg_spe)
print('bs_train', bs_train, 'bs_test', bs_test, 'sen', avg_sen)
print('bs_train', bs_train, 'bs_test', bs_test, 'f1', avg_f1)
print('bs_train', bs_train, 'bs_test', bs_test, 'auc', avg_auc)
print('bs_train', bs_train, 'bs_test', bs_test, 'pre', avg_pre)
print('bs_train', bs_train, 'bs_test', bs_test, 'recall', avg_recall)
print('nc_fle', 'bs_train:', bs_train, 'bs_test:', bs_test, 'lr', lr)
print('标准差:', 'acc_std', acc_std,  'spe_std', spe_std, 'sen_std', sen_std, 'f1_std', f1_std, 'auc_std', auc_std, 'pre_std', pre_std, 'recall_std', recall_std)
