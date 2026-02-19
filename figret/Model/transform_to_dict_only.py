import torch
import sys
import os


sys.path.append('xxx/rapidAIsim_ai4topo/figret')


for file in os.listdir('.'):
    if file.endswith('.pt'):
        model = torch.load(file)
        torch.save(model.state_dict(), file.replace('.pt', '.pth'))
