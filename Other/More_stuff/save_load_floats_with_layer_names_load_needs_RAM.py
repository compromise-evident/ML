# Untested. Only load crashes if not enough RAM for big models.


# Saves model to file. (No crash.)
with open('Model/Model.txt', 'w') as f:
	for key, value in model.state_dict().items():
		f.write(f"{key}: {value.float().tolist()}\n")

# Loads model from file. (Crash if not enough RAM.)
loaded = {}
with open('Model/Model.txt', 'r') as f:
	for line in f:
		key, value_str = line.strip().split(': ')
		values = torch.FloatTensor(eval(value_str))
		loaded[key] = values
model.load_state_dict(loaded)
