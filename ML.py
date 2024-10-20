# ML 2.0.0 - AI simplified. Included training-data is ezMNIST. Use it to test     Run it: "apt install geany python3-torch". Open the .py in Geany.
#            model generalization then replace it. PyTorch, 1 file, 80 lines.     Replace "python" with "python3" in Geany's execute command. F5 to run.

import torch
import torch.nn as nn
import torch.optim as optim

# YOUR CONTROLS - UNLIMITED:
longest = 784 # Longest data string in train.txt, test.txt, cognize.txt  (safe.)  input layer
classes =  10 # Number of different labels (2 = labels 0,1. 500 = labels 0-499.)  output  layer
depth   =   2 # Number of hidden layers  (the active brain parts of your model.)  n hidden layers
width   =  16 # Number of neurons per hidden layer (wide = attentive to detail.)  hidden layer size
compute ='cpu'# 'cuda' for NVIDIA GPU, 'mps' for Mac GPU ('cuda:0', 'cuda:1'...)  hardware to harness

print("\n(1) Model   (Create a new model and save it as one file.)")
print(  "(2) Train   (Train & test model on train.txt & test.txt.)")
print(  "(3) Test    (See only testing on test.txt - no training.)")
print(  "(4) Use     (Classify unlabeled cognize.txt - no spaces.)")
o = int(input("\nOption: "))

# Needed to create/save/load model.
model = nn.Sequential(); model.add_module('input',       nn.Linear(longest, width  )); model.add_module('relu1',     nn.ReLU());
for a in range(depth):   model.add_module(f'hidden_{a}', nn.Linear(  width, width  )); model.add_module(f'relu_{a}', nn.ReLU());
model.add_module                         ('output',      nn.Linear(  width, classes)); normalized = [0.0] * longest;

if o == 1: # Model___________________________________________________________________________________________________________________________________________________
	torch.save(model.state_dict(), 'Model.pth'); print("\nModel.pth saved with hidden layers:  ", depth, "deep,", width, "wide.")      # Saves model to file.

if o == 2: # Train___________________________________________________________________________________________________________________________________________________
	model.load_state_dict(torch.load('Model.pth', map_location = compute))                                                             # Loads model from file.
	with open('training-data/train.txt', 'rb') as f: total_training_data_items = f.read().count(10)                                    # Number of items to train on.
	in_stream = open('training-data/train.txt', 'r')
	criterion = nn.CrossEntropyLoss(); optimizer = optim.SGD(model.parameters(), lr = 0.01); model.train(); print("\n", end = '');
	for a in range(total_training_data_items):
		print("Training on train.txt line", (a + 1), "of", total_training_data_items)
		line = in_stream.readline().split(); target_data = torch.tensor([int(line[0])]);                                               # Forces classification.
		normalized[:] = [0.0] * longest; length = len(line[1]);                                                                        # Data to be classified.
		if length > longest: length = longest
		for b in range(length):
			if line[1][b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		optimizer.zero_grad(); outputs = model(input_data); loss = criterion(outputs, target_data); loss.backward(); optimizer.step(); # Uses & updates model.
	in_stream.close(); torch.save(model.state_dict(), 'Model.pth');                                                                    # Saves updated model.

if o == 3 or o == 2: # Test__________________________________________________________________________________________________________________________________________
	model.load_state_dict(torch.load('Model.pth', map_location = compute))                                                             # Loads model from file.
	with open('training-data/test.txt', 'rb') as f: total_testing_data_items = f.read().count(10)                                      # Number of items to test on.
	model.eval(); misclassified = 0; print("\n", end = '');
	in_stream = open('training-data/test.txt', 'r'); out_stream = open('results.txt', 'w');
	for a in range(total_testing_data_items):
		print("Testing on test.txt line", (a + 1), "of", total_testing_data_items)
		line = in_stream.readline().split(); expected_classification = int(line[0]);                                                   # Expected classification.
		normalized[:] = [0.0] * longest; length = len(line[1]);                                                                        # Data to be classified.
		if length > longest: length = longest
		for b in range(length):
			if line[1][b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);                                     # Uses model.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");                                               # Saves classification.
		if classification != expected_classification: misclassified += 1                                                               # Checks if misclassified.
	in_stream.close(); out_stream.close();
	print("\nMisclassifies", misclassified, "out of", total_testing_data_items, "(see results.txt)\n")
	percent_correct = format((((total_testing_data_items - misclassified) / total_testing_data_items) * 100), ".15f")
	print(percent_correct, end = '% correct.')

if o == 4: # Use_____________________________________________________________________________________________________________________________________________________
	model.load_state_dict(torch.load('Model.pth', map_location = compute))                                                             # Loads model from file.
	with open('cognize.txt', 'rb') as f: total_real_world_items = f.read().count(10)                                                   # Number of items to cognize.
	model.eval(); print("\n", end = '');
	in_stream = open('cognize.txt', 'r'); out_stream = open('results.txt', 'w');
	for a in range(total_real_world_items):
		print("Classifying cognize.txt line", (a + 1), "of", total_real_world_items)
		normalized[:] = [0.0] * longest; line = in_stream.readline(); length = (len(line) - 1);                                        # Data to be classified.
		if length > longest: length = longest
		for b in range(length):
			if line[b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);                                     # Uses model.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");                                               # Saves classification.
	in_stream.close(); out_stream.close(); print("\nSee results.txt");
