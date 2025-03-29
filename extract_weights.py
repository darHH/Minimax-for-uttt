import torch
from uttt_model import UTTTEvaluator

# torch.set_printoptions(profile="full")
model = UTTTEvaluator()
model.load_state_dict(torch.load("trained_uttt_model.pth"))
print(model.state_dict())