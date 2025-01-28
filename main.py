import torch
from model import ResNet18
from torchvision import transforms
import shap
from PIL import Image
import matplotlib.pyplot as plt
from masking import ChannelMeanMask, channel_mean_masking
import numpy as np

# torch.set_grad_enabled(False)

model = ResNet18()
transform = transforms.Compose(
    [
        transforms.Resize((256, 256), Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image = Image.open("test_image.jpg")
input_tensor = transform(image).unsqueeze(0)
# Plot input tensor
plt.imshow(input_tensor.squeeze().permute(1, 2, 0))
plt.show()


output = model(input_tensor)
print(f"input.shape = {input_tensor.shape}")
print(f"output.shape = {output.shape}")
print(f"output argmax = {output.argmax()}")

# Use SHAP with custom masking function
# explainer = shap.Explainer(model, masker=channel_mean_masking, max_evals=301057)
# explainer = shap.DeepExplainer(model)
input_tensor_mean_per_channel = torch.mean(input_tensor, dim=(2, 3), keepdim=True)
background_data = input_tensor_mean_per_channel * torch.ones_like(input_tensor)
print(f"input_tensor_mean_per_channel.shape = {input_tensor_mean_per_channel.shape}")
print(f"background_data.shape = {background_data.shape}")
print(model(background_data))

X, y = shap.datasets.imagenet50()
X = torch.tensor(X).permute(0, 3, 1, 2)
print(f"X.shape = {X.shape}")


def f(x: np.ndarray):
    x = x.reshape(1, 3, 224, 224)
    x = torch.tensor(x)
    return model(x).numpy().flatten()


explainer = shap.DeepExplainer(
    model,
    background_data,
)
shap_values = explainer(input_tensor, check_additivity=False)
print(shap_values)

# Plot SHAP values
# shap_numpy = [np.transpose(s, (1, 2, 0)) for s in shap_values.values]
# shap.image_plot(shap_numpy, -input_tensor.squeeze().permute(1, 2, 0).numpy())
# plt.show()
