# ML - AI unfucked for all to use. Included training-data contains              Run it: "apt install geany python3-torch". Open the .py in Geany.
#      ezMNIST. Use it to test model generalization then replace it.            Replace "python" with "python3" in Geany's execute command. F5 to run.
#      Uses PyTorch, single file, under 200 lines.


# Version 1.0.0
import torch
import torch.nn as nn
import torch.optim as optim

# YOUR CONTROLS; UNLIMITED,
# GET NEW MODEL PER CHANGE:
longest = 784 # Longest data string in train.txt, test.txt, cognize.txt  (safe.)
classes =  10 # Number of different labels (2 = labels 0,1. 500 = labels 0-499.)
depth   =   3 # Number of hidden layers  (the active brain parts of your model.)
width   = 100 # Number of neurons per hidden layer (wide = attentive to detail.)

print("\n(1) Model   (Create a new model and save it as one file.)")
print(  "(2) Train   (Train & test model on train.txt & test.txt.)")
print(  "(3) Test    (See only testing on test.txt - no training.)")
print(  "(4) Use     (Classify unlabeled cognize.txt - no spaces.)")
option = int(input("\nOption: "))

# This is needed to create/save/load model.
model = nn.Sequential(); model.add_module('input',       nn.Linear(longest, width  )); model.add_module('relu1',     nn.ReLU());
for a in range(depth):   model.add_module(f'hidden_{a}', nn.Linear(  width, width  )); model.add_module(f'relu_{a}', nn.ReLU());
model.add_module                         ('output',      nn.Linear(  width, classes))

#_________________________________________________________Model_________________________________________________________
if option == 1:
	# Saves model to file.
	torch.save(model.state_dict(), 'Model.pth')
	print("\nModel.pth saved with hidden layers:  ", depth, "deep,", width, "wide.")

#_________________________________________________________Train_________________________________________________________
elif option == 2:
	# Loads model from file.
	model.load_state_dict(torch.load('Model.pth', map_location='cpu'))
	
	# Gets number of items to train on.
	with open('training_data/train.txt', 'rb') as f: total_training_data_items = f.read().count(10)
	
	# Trains model.
	print("\n", end='')
	in_stream = open('training_data/train.txt', 'r')
	criterion = nn.CrossEntropyLoss(); optimizer = optim.SGD(model.parameters(), lr = 0.01); model.train();
	for a in range(total_training_data_items):
		print("Training on train.txt line", (a + 1), "of", total_training_data_items)
		
		# Gets label and data.
		line = in_stream.readline(); parts = line.split(); label = int(parts[0]); raw_data = parts[1]; length = len(raw_data);
		if length > longest: length = longest # If data is longer than "longest", ML will use the first "longest" characters of that data.
		
		# Force classification.
		target_data = torch.tensor([label]);
		
		# Data to be classified.
		normalized = [0.0] * longest;
		for b in range(length):
			if raw_data[b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		
		# Uses & updates model.
		optimizer.zero_grad()                  # Zeros gradients.
		outputs = model(input_data)            # Forward pass.
		loss = criterion(outputs, target_data) # Computes loss.
		loss.backward()                        # Backward pass.
		optimizer.step()                       # Updates model constructively.
	in_stream.close()
	
	# Saves updated model to file (overwrites.)
	torch.save(model.state_dict(), 'Model.pth')
	
	# Gets number of items to test on.
	with open('training_data/test.txt', 'rb') as f: total_testing_data_items = f.read().count(10)
	
	# Tests model.
	print("\n\n")
	model.eval()
	misclassified = 0
	in_stream = open('training_data/test.txt', 'r')
	out_stream = open('results.txt', 'w')
	for a in range(total_testing_data_items):
		print("Testing on test.txt line", (a + 1), "of", total_testing_data_items)
		
		# Gets label and data.
		line = in_stream.readline(); parts = line.split(); label = int(parts[0]); raw_data = parts[1]; length = len(raw_data);
		if length > longest: length = longest # If data is longer than "longest", ML will use the first "longest" characters of that data.
		
		# Expected classification.
		expected_classification = label
		
		# Data to be classified.
		normalized = [0.0] * longest;
		for b in range(length):
			if raw_data[b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		
		# Uses model.
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);
		
		# Appends classification to file results.txt.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");
		
		# Checks if misclassified.
		if classification != expected_classification: misclassified += 1
	in_stream.close()
	out_stream.close()
	
	print("\nMisclassifies", misclassified, "out of", total_testing_data_items, "(see results.txt)\n")
	percent_correct = format((((total_testing_data_items - misclassified) / total_testing_data_items) * 100), ".15f")
	print(percent_correct, end='% correct.')

#__________________________________________________________Test_________________________________________________________
elif option == 3:
	# Loads model from file.
	model.load_state_dict(torch.load('Model.pth', map_location='cpu'))
	
	# Gets number of items to test on.
	with open('training_data/test.txt', 'rb') as f: total_testing_data_items = f.read().count(10)
	
	# Tests model.
	print("\n", end='')
	model.eval()
	misclassified = 0
	in_stream = open('training_data/test.txt', 'r')
	out_stream = open('results.txt', 'w')
	for a in range(total_testing_data_items):
		print("Testing on test.txt line", (a + 1), "of", total_testing_data_items)
		
		# Gets label and data.
		line = in_stream.readline(); parts = line.split(); label = int(parts[0]); raw_data = parts[1]; length = len(raw_data);
		if length > longest: length = longest # If data is longer than "longest", ML will use the first "longest" characters of that data.
		
		# Expected classification.
		expected_classification = label
		
		# Data to be classified.
		normalized = [0.0] * longest;
		for b in range(length):
			if raw_data[b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		
		# Uses model.
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);
		
		# Appends classification to file results.txt.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");
		
		# Checks if misclassified.
		if classification != expected_classification: misclassified += 1
	in_stream.close()
	out_stream.close()
	
	print("\nMisclassifies", misclassified, "out of", total_testing_data_items, "(see results.txt)\n")
	percent_correct = format((((total_testing_data_items - misclassified) / total_testing_data_items) * 100), ".15f")
	print(percent_correct, end='% correct.')

#__________________________________________________________Use__________________________________________________________
elif option == 4:
	# Loads model from file.
	model.load_state_dict(torch.load('Model.pth', map_location='cpu'))
	
	# Gets number of items to cognize.
	with open('cognize.txt', 'rb') as f: total_real_world_items = f.read().count(10)
	
	# Tests model.
	print("\n", end='')
	model.eval()
	in_stream = open('cognize.txt', 'r')
	out_stream = open('results.txt', 'w')
	for a in range(total_real_world_items):
		print("Classifying cognize.txt line", (a + 1), "of", total_real_world_items)
		
		# Gets data.
		raw_data = in_stream.readline(); length = len(raw_data); length -= 1;
		if length > longest: length = longest # If data is longer than "longest", ML will use the first "longest" characters of that data.
		
		# Data to be classified.
		normalized = [0.0] * longest;
		for b in range(length):
			if raw_data[b] == '@': normalized[b] = 1.0
		input_data = torch.tensor(normalized).view(1, longest)
		
		# Uses model.
		with torch.no_grad(): outputs = model(input_data); _, predictions = torch.max(outputs, 1);
		
		# Appends classification to file results.txt.
		classification = predictions[0].item(); out_stream.write(f"{classification}\n");
	in_stream.close()
	out_stream.close()
	print("\nSee results.txt")

else: print("\nInvalid.")
