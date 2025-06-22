# XOR Neural Network

## A fully functional neural network written in python from scratch that learns the output of an XOR function 

This project was buit with no external libraries at all using only core python, building a neural network from the ground up. This project:
- Implements a value class for autograd (automatic differentiation)
- Uses multiple layers of neurons with tanh activation
- Trains a multi-layer perceptron to solve the XOR problem
- Includes forward and backward passes, manual gradients, and weight updates
- Prints the training as it is happening real time every 200 epochs (one epoch is one runthrough through the whole dataset)
- Outputs the final predictions for each XOR function after 4000 epochs 

## How to run 
1. Make sure python is installed on your device (You can verify this by running the command "python --version" in your terminal/command prompt")

2. Download the script
- If you have git installed, run:
  git clone https://github.com/Supposeduck30/xor-neural-network.git
- Or, click the green code button and press "download zip, and then extract it 

3. Inside of the directory, open the file named "NeuralNetwork.py"

4. ALTERNATIVE - You can paste the code into an online Python compiler

## How to tweak this project for your own use 
1. Fork the repository
   
2. Clone the fork
   
3. Make your changes to the code
   
4. Commit and push your changes to the fork
   
5. OPTIONAL - Create a pull request if you want the main repository to change the code with what you changed 

## How it works 
#### 1. Inputs go in
   - The model takes two numbers as an input (like [0, 1] or [1, 0])

#### 2. The "neurons" do math
   - It runs those inputs through multiple layers of math formulas called neurons
   - Each neuron adds the inputs in a unique way and then squashes the result into a number between -1 and 1 with the tanh function

#### 3. It guesses an answer

#### 4. It checks how wrong the guess is
   - It compares it's answer to the right one (called loss)
  
#### 5. It then changes its math to be a little more accurate

#### 6. It does this loop 4000 times

#### 7. It outputs its final prediction
   - This prediction will sometimes not be spot on, as the model still might need more training

- Each connection between neurons is random every single time, which helps the network learn better

## Resources
### https://www.ibm.com/think/topics/neural-networks (Neural Networks Explained)

## Known Issues 
- Due to the random seed, the results will be different every time the code is run
- Some predictions may be in the negatives but still close to 0 (this is due to the tanh function)
