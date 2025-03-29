import torch
from uttt_model import UTTTEvaluator

torch.set_printoptions(profile="full")

model = UTTTEvaluator()
model.load_state_dict(torch.load("trained_uttt_model.pth"))

with open("model_weights.txt", "w") as file:
    for param_name, param_tensor in model.state_dict().items():
        file.write(f"{param_name}:\n{param_tensor}\n\n")