import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

 
import argparse
from tensorboardX import SummaryWriter
from network import feature_net



#参数设置
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--dataset_dir', default='../dogs_vs_cats/')
parser.add_argument('--checkpoint_dir', default='./checkpoint')
parser.add_argument('--record_dir', default='./record')
parser.add_argument('--log_dir', default='./log')
parser.add_argument('--pre_epoch', default=0, help='begin epoch')
parser.add_argument('--total_epoch', default=1, help='time for ergodic')
parser.add_argument('--model', default='inceptionv3', help='model for training')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--pre_model', default=False, help='use pre-model')#恢复训练时的模型路径
parser.add_argument('--batch_size', default=32)
parser.add_argument('--CenterCropSize', default=299)
args = parser.parse_args()


#定义使用模型
model = args.model
#使用gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#图片导入
path = args.dataset_dir
transform = transforms.Compose([transforms.CenterCrop(args.CenterCropSize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_image = {x:datasets.ImageFolder(root=os.path.join(path, x),
                                     transform=transform)
              for x in ["train", "val"]}
data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
                     for x in ["train", "val"]}

classes = data_image["train"].classes#按文件夹名字分类
classes_index = data_image["train"].class_to_idx#文件夹类名所对应的键值
print(classes)
print(classes_index)
#打印训练验证集
print("train data set:", len(data_image["train"]))
print("val data set:", len(data_image["val"]))

image_train, label_train = next(iter(data_loader_image["train"]))
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = torchvision.utils.make_grid(image_train)#把batch_size张的图片拼成一个图片
#print(img.shape)#[3, 228, 906]
img = img.numpy().transpose((1, 2, 0))#本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
#print(img.shape)
img = img*std + mean#(228, 906, 3)范围由(-1, 1)变成(0, 1)

print([classes[i] for i in label_train])#打印image_train图像中对应的label_train，也就是图像的类型
plt.imshow(img)#mshow能显示数据归一化到0到1的图像
#plt.show()

#分类器工厂
classifier_factory = {
    'vgg': 512,
    'resnet50': 2048,
    'inceptionv3': 2048
}


#构建网络
use_model = feature_net(model, dim=classifier_factory[args.model], n_classes=2)
for parma in use_model.feature.parameters():
    parma.requires_grad = False

for index, parma in enumerate(use_model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

if use_cuda:
    use_model = use_model.to(device)
#损失函数
loss = torch.nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.Adam(use_model.classifier.parameters())
#打印网络
print(use_model)


#使用预训练模型
if args.pre_model:
    print("Resume from checkpoint...")
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found'
    state_path = args.checkpoint_dir + '/' + args.model + '/ckpt.t7'
    state = torch.load(state_path)
    use_model.load_state_dict(state['state_dict'])
    best_test_acc = state['acc']
    pre_epoch = state['epoch']
else:
    #定义最优的测试准确率
    best_test_acc = 0
    pre_epoch = args.pre_epoch


if __name__ == '__main__':
    # 没有就创建checkpoint文件夹
    if not os.path.exists(args.checkpoint_dir + '/' + args.model):
        os.makedirs(args.checkpoint_dir + '/' + args.model)
        print("Checkpoint directory has been created:", str(args.checkpoint_dir + '/' + args.model))
    else:
        print("Checkpoint directory is already existed:", str(args.checkpoint_dir + '/' + args.model))
    if not os.path.exists(args.outf + '/' + args.model):
        os.makedirs(args.outf + '/' + args.model)
        print("Model directory has been created:", str(args.outf + '/' + args.model))
    else:
        print("Model directory is already existed:", str(args.outf + '/' + args.model))
    total_epoch = args.total_epoch
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.model)
    print("Start Training, {}...".format(model))
    record_path = args.record_dir
    acc_path = args.record_dir + '/' + args.model + '/acc.txt'
    log_path = args.record_dir + '/' + args.model + '/log.txt'
    if not os.path.exists(args.record_dir + '/' + args.model):
        os.makedirs(args.record_dir + '/' + args.model)
        print("acc.txt and log.txt will be recorded into:", str(args.record_dir + '/' + args.model))
    else:
        print("record directory is already existed:", str(args.record_dir + '/' + args.model))
    with open(acc_path, "w") as acc_f:
        with open(log_path, "w") as log_f:
            start_time = time.time()
            
            for epoch in range(pre_epoch, total_epoch):
                
                print("epoch{}/{}".format(epoch, total_epoch))
                print("-"*10)
                #开始训练
                use_model.train()
                # print(use_model)
                #初始化
                sum_loss = 0.0
                accuracy = 0.0
                total = 0

                for i, data in enumerate(data_loader_image["train"]):
                    image, label = data
                    if use_cuda:
                        image, label = Variable(image.to(device)), Variable(label.to(device))
                    else:
                        image, label = Variable(image), Variable(label)

                    optimizer.zero_grad()
                    label_prediction = use_model(image)

                    _, prediction = torch.max(label_prediction.data, 1)
                    total += label.size(0)
                    current_loss = loss(label_prediction, label)

                    current_loss.backward()
                    optimizer.step()

                    sum_loss += current_loss.item()
                    accuracy += torch.sum(prediction == label.data)

                    if total % 5 == 0:
                        print("total {}, train loss:{:.4f}, train accuracy:{:.4f}".format(
                            total, sum_loss/total, 100 * accuracy/total))
                        #写入日志
                        log_f.write("total {}, train loss:{:.4f}, train accuracy:{:.4f}".format(
                            total, sum_loss/total, 100 * accuracy/total))
                        log_f.write('\n')
                        log_f.flush()

                #写入tensorboard
                writer.add_scalar('loss/train', sum_loss / (i + 1), epoch)
                writer.add_scalar('accuracy/train', 100. * accuracy / total, epoch)
                #每一个epoch测试准确率
                print("Waiting for test...")
                #在上下文环境中切断梯度计算，在此模式下，每一步的计算结果中requires_grad都是False，即使input设置为requires_grad=True
                with torch.no_grad():
                    accuracy = 0
                    total = 0
                    for data in data_loader_image["val"]:
                        use_model.eval()

                        image, label = data
                        if use_cuda:
                            image, label = Variable(image.to(device)), Variable(label.to(device))
                        else:
                            image, label = Variable(image), Variable(label)

                        label_prediction = use_model(image)

                        _, prediction = torch.max(label_prediction.data, 1)
                        total += label.size(0)
                        accuracy += torch.sum(prediction == label.data)
                
                    #输出测试准确率
                    print('测试准确率为: %.3f%%' % (100 * accuracy / total))
                    acc = 100. * accuracy / total

                    #写入tensorboard
                    writer.add_scalar('accuracy/test', acc,epoch)
                    
                    #将测试结果写入文件

                    model_path = args.outf + '/' + args.model + '/net_%3d.pth' % (epoch + 1)
                    torch.save(use_model.state_dict(), model_path)
                    print("Model has been saved in:", model_path)
                    acc_f.write("epoch = %03d, accuracy = %.3f%%" % (epoch + 1, acc))
                    acc_f.write('\n')
                    acc_f.flush()

                    #记录最佳的测试准确率
                    if acc > best_test_acc:
                        #存储状态
                        state = {
                            'state_dict': use_model.state_dict(),
                            'acc': acc,
                            'epoch': epoch + 1,
                        }
                        state_path = args.checkpoint_dir + '/' + args.model + '/ckpt.t7'
                        torch.save(state, state_path)
                        print("Best model state has been saved in:", state_path)
                        best_test_acc = acc
                        #写入tensorboard
                        writer.add_scalar('best_accuracy/test', best_test_acc, epoch)
            
            end_time = time.time() - start_time
            print("training time is:{:.0f}m {:.0f}s".format(end_time // 60, end_time % 60))
            writer.close()
