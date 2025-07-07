from torchinfo import summary

from models.twcnn import TWCNN
from models.twinception import TWInception
from models.twinception_small import TWInceptionSmall

def model_summary(model):
    input_size = (16, 200, 15, 15)
    summary(model, input_size=input_size, col_names=["output_size", "num_params", "kernel_size", "mult_adds"])

model = TWCNN(200, 16)
model_summary(model)

model = TWInception(200, 16)
model_summary(model)

model = TWInceptionSmall(200, 16)
model_summary(model)

