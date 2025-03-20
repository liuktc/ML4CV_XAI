# Attribution path: "{image_index}.{layer}.{attribution}.{upsample}.png"
# Image path: "{image_index}.png"
# Attribution metrics: "{image_index}.{layer}.{attribution}.{upsample}.csv"

from utils import _DeepLiftShap, _GradCAMPlusPlus
from data import PascalVOC2007
from models import vgg11_PascalVOC, vgg_preprocess
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vgg11_PascalVOC()
model.to(device)
# Load the pretrained weights
model.load_state_dict(torch.load("VGG11_PascalVOC.pt", map_location=device))
model.eval()

test_data = PascalVOC2007("test", transform=vgg_preprocess)

for image_index in range(10):
    image, label = test_data[image_index]
    print(image.shape, label.shape)
    break
