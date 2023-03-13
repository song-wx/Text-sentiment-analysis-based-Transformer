import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


def train(model, trainloader, testloader, epoch_nums, learning_rate=0.1):
    #定义Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #定义交叉熵损失函数
    loss_fun = F.cross_entropy
    # 训练，测试的损失记录
    train_loss_record = list()
    test_loss_record = list()
    # 训练， 测试的准确率记录
    train_acc_record = list()
    test_acc_record = list()

    """开始训练，进行epoch_nums个epoch"""
    for epoch in tqdm(range(epoch_nums)):
        ##进入新一轮epoch
        model.train() ##模型进入训练模式
        for i, batch in enumerate(trainloader):
            x, y = batch ##取一个batch 
            y = y.cuda()
            x = x.cuda()
            pred = model(x).cuda() ##计算输出
            optimizer.zero_grad() ##梯度重置
            loss = loss_fun(pred, y).cuda() ##计算损失
            loss.backward() ##反向传播梯度
            optimizer.step() ##更新参数
        
        model.eval() ##进入评估模式
        ##测试本轮epoch训练情况
        ##准确率
        train_acc = list()
        test_acc = list()
        ##损失
        train_loss = list()
        test_loss = list()
        ##计算在训练集中的损失和准确率
        for i, batch in enumerate(trainloader):
            x, y = batch  # 取一个batch y的维度为[batchsize, 1]
            y = y.cuda()
            x = x.cuda()
            pred = model(x).cuda()  # 计算输出,pred维度为[batch,5]
            loss = loss_fun(pred, y).cuda()    # 损失值计算
            train_loss.append(loss.item())   # 记录本次损失值
            _, y_pre = torch.max(pred, -1) ##最大值对应的类别，维度为[batch,1]
            # 计算本batch准确率
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float))).item()
            train_acc.append(acc)

        ##计算在测试集中的损失和准确率
        for i, batch in enumerate(testloader):
            x, y = batch  # 取一个batch
            y = y.cuda()
            x = x.cuda() 
            pred = model(x).cuda()  # 计算输出
            loss = loss_fun(pred, y).cuda()    # 损失值计算
            test_loss.append(loss.item())   # 记录本次损失值
            _, y_pre = torch.max(pred, -1)
            # 计算本batch准确率
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float))).item()
            test_acc.append(acc)

        ##此次epoch的平均accuracy
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_test_acc = sum(test_acc) / len(test_acc)

        ##此次epoch的平均loss
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_test_loss = sum(test_loss) / len(test_loss)

        ##将acc和loss保存
        test_acc_record.append(avg_test_acc)
        train_acc_record.append(avg_train_acc)
        train_loss_record.append(avg_train_loss)
        test_loss_record.append(avg_test_loss)

        ##输出loss和acc信息
        print(f"---------- Epoch {epoch + 1} ----------")
        print("Train loss:", avg_train_loss)
        print("Test loss:", avg_test_loss)
        print("Train accuracy:", avg_train_acc)
        print("Test accuracy:", avg_test_acc)

    return train_loss_record, test_loss_record, train_acc_record, test_acc_record



