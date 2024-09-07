import os
import pandas as pd
import shutil
import zipfile
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义工作目录
working_dir = '/kaggle/working/'

# 确保目录存在
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# 解压函数
def unzip_file(zip_path, dest_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
        print(f'解压完成: {zip_path} 到 {dest_folder}')

# 解压文件路径
test_zip_path = '/kaggle/input/street-view-getting-started-with-julia/test.zip'
train_zip_path = '/kaggle/input/street-view-getting-started-with-julia/train.zip'

# 解压 test.zip 和 train.zip
unzip_file(test_zip_path, working_dir)
unzip_file(train_zip_path, working_dir)

# 读取 sampleSubmission.csv 和 trainLabels.csv
sample_submission = pd.read_csv('/kaggle/input/street-view-getting-started-with-julia/sampleSubmission.csv')
labels_df = pd.read_csv('/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv')

# 划分训练集和验证集
train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['Class'], random_state=42)

# 保存到CSV
output_dir = working_dir
train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
data_dir = working_dir
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
}

# 创建数据加载器
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}