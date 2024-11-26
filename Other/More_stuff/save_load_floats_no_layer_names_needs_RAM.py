# Functional. Crashes if not enough RAM for big models.


# Saves model to file.
with open('Model/Model.txt', 'w') as f:
	for param in model.parameters():
		param_data = param.data.view(-1).tolist()
		f.write(' '.join(map(str, param_data)) + '\n')

# Loads model from file.
with open('Model/Model.txt', 'r') as f:
	for param in model.parameters():
		param_data = list(map(float, f.readline().strip().split()))
		param.data.copy_(torch.tensor(param_data).view_as(param))
