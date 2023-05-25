from model import GeneratorResNet
import torch
import torch.nn as nn


Gen_BA = nn.DataParallel(GeneratorResNet((3,256,256), 9))
checkpoint = torch.load("checkpoint\checkpoint-50\CycleGan_VanGogh_Checkpoint.pt", map_location='cpu')
if checkpoint is not None:
    print("Loading checkpoint...")
    Gen_BA.load_state_dict(checkpoint['Gen_BA'])
    print("Successfully loaded checkpoint.")
    Gen_BA_scripted = torch.jit.script(Gen_BA.module) # Export to TorchScript
    Gen_BA_scripted.save('CycleGAN_Generator_50.pt') # Save