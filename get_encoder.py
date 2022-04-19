import tensorflow as tf
from improved_nightdrive.segmentation.models import UNet_MobileNetV2, get_encoder_unetmobilenetv2, DeeplabV3, get_encoder_deeplabv3

# model = UNet_MobileNetV2(224, 19)
# model.load_weights("./results/sweep/unetmobilenetv2_day_only/models/unetmobilenetv2_at_49")
# new = get_encoder_unetmobilenetv2(model)
# new.summary()

model = DeeplabV3(224, 19)
model.load_weights("./results/sweep/deeplabv3_day_only/models/deeplabv3_at_49")
new = get_encoder_deeplabv3(model)
new.summary()