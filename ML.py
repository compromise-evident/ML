# ML 2.0.2 - AI simplified in 1 file, 80 lines. Verify that your model can generalize on the given training-            Run it: "apt install geany python3-torch". Open the .py in Geany.
#            data by scoring well. Then replace the data with your own (label & string per line in text file.)          Replace "python" with "python3" in Geany's execute command. F5 to run.

import torch
import torch.nn as nn
import torch.optim as optim

# YOUR CONTROLS - UNLIMITED - GET NEW MODEL PER CHANGE:
longest = 784 # Longest data string in train.txt, test.txt, cognize.txt  (safe.)  input layer
classes =  10 # Number of different labels (2 = labels 0,1. 500 = labels 0-499.)  output  layer
depth   =   2 # Number of hidden layers  (the active brain parts of your model.)  n hidden layers
width   =  16 # Number of neurons per hidden layer (wide = attentive to detail.)  hidden layer size

print("\n(1) Model   (Create a new model and save it as one file.)")
print(  "(2) Train   (Train & test model on train.txt & test.txt.)")
print(  "(3) Test    (See only testing on test.txt - no training.)")
print(  "(4) Use     (Classify unlabeled cognize.txt - no spaces.)"); o = int(input("\nOption: "));

model = nn.Sequential(); model.add_module('input',       nn.Linear(longest, width  )); model.add_module('relu1',     nn.ReLU());
for a in range(depth):   model.add_module(f'hidden_{a}', nn.Linear(  width, width  )); model.add_module(f'relu_{a}', nn.ReLU());
model.add_module                         ('output',      nn.Linear(  width, classes)); normalized = [0.0] * longest;

if o == 1: # Model___________________________________________________________________________________________________________________________________________________
	torch.save(model.state_dict(), 'Model.pth'); print("\nModel.pth saved with hidden layers:  ", depth, "deep,", width, "wide.")      # Saves model to file.

if o == 2: # Train___________________________________________________________________________________________________________________________________________________
	model.load_state_dict(torch.load('Model.pth', map_location = 'cpu'))                                                               # Loads model from file.
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
	model.load_state_dict(torch.load('Model.pth', map_location = 'cpu'))                                                               # Loads model from file.
	with open('training-data/test.txt', 'rb') as f: total_testing_data_items = f.read().count(10)                                      # Number of items to test on.
	misclassified = 0; off_by_summation = 0; model.eval(); print("\n", end = '');
	in_stream = open('training-data/test.txt', 'r'); out_stream = open('results.txt', 'w'); out_xtra = open('results_extra.txt', 'w');
	for a in range(total_testing_data_items):
		print("Testing on test.txt line", (a + 1), "of", total_testing_data_items)
		line = in_stream.readline().split(); expected_class = int(line[0]);                                                            # Expected classification.
		normalized[:] = [0.0] * longest; length = len(line[1]);                                                                        # Data to be classified.
		if length > longest: length = longest
		for b in range(length):
			if line[1][b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);                                     # Uses model.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");                                               # Saves classification.
		if classification != expected_class: misclassified += 1; off_by_summation += (abs(expected_class - classification));           # Checks if misclassified.
		if classification == expected_class: out_xtra.write(f"{classification} (OK\n")
		else: out_xtra.write(f"{classification} ({expected_class}  was the label - off by {abs(expected_class - classification)}\n")
	in_stream.close(); out_stream.close(); out_xtra.close();
	print("\n\n\n", format((((total_testing_data_items - misclassified) / total_testing_data_items) * 100), ".15f"), end = "% correct")
	print(" (misclassifies", misclassified, "out of", total_testing_data_items, end = ")\n\n")
	print(f"Off by {off_by_summation / misclassified} on average (see results_extra.txt)")

if o == 4: # Use_____________________________________________________________________________________________________________________________________________________
	model.load_state_dict(torch.load('Model.pth', map_location = 'cpu'))                                                               # Loads model from file.
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
