import torch
import torchvision.models as models

model = models.resnet18(num_classes=42)

# 2. 加载你训练保存的权重
checkpoint = torch.load("best.pt", map_location="cpu")
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict)

model.eval()

# 3. 虚拟输入：batch=1，3通道，224×224（和训练保持一致）
dummy_input = torch.randn(1, 3, 224, 224)

# 4. 导出ONNX模型
torch.onnx.export(
    model,
    dummy_input,
    "houseplant.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"]
)

print("模型导出完成！")