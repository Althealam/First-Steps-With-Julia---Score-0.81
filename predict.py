import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from data import dataloaders, device

# 加载模型
model = get_model(62)
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 获取测试集图像的完整路径列表
test_dir = '/kaggle/working/test'
image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]
num_images = len(image_paths)
batch_size = 32

predictions = []

# 分批处理
for i in range(0, num_images, batch_size):
    batch_paths = image_paths[i:i + batch_size]
    images = []

    for img_path in batch_paths:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image)
        images.append(input_tensor)

    batch_tensor = torch.stack(images)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted_indices = torch.max(outputs, 1)

    for img_path, predicted_idx in zip(batch_paths, predicted_indices):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        predictions.append((img_id, predicted_idx.item()))

# 保存预测结果
output_df = pd.DataFrame(predictions, columns=['ID', 'Predicted'])
output_df.to_csv('submission.csv', index=False)