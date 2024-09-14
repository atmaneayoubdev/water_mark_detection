import torch

# Print the number of GPUs detected
print("Number of GPUs detected:", torch.cuda.device_count())

# Print the name of each GPU detected
for i in range(torch.cuda.device_count()):
    print("GPU", i, ":", torch.cuda.get_device_name(i))
