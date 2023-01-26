import torch
import torchvision.models as models
from dcn.deform_conv import DCN

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Replace the standard convolutional layers with deformable convolutional layers
for idx, layer in enumerate(model.modules()):
    if isinstance(layer, torch.nn.Conv2d):
        dcn = DCN(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, deformable_groups=1)
        model.add_module(str(idx), dcn)

# Print the model architecture
print(model)
