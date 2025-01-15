import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, datasets_test):
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    pro_all = []
    model.eval()  # 将模型改为预测模式
    for feature1, adj1, feature2, adj2 ,label in datasets_test:
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
        # 记录误差
        eval_loss += float(loss)
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / feature1.shape[0]
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(out[:, 1].cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)

    return eval_loss, eval_acc, eval_acc_epoch, specificity, sensitivity, f1, my_auc,  precision, recall, labels_all, pro_all