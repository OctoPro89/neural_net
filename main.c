#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Activation functions
typedef double (*ActivateFunc)(double);
typedef double (*ActivateDerivative)(double);

// could add ReLU
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { return x * (1.0 - x); }

// neural net
typedef struct
{
	double* weights;
	double bias;
	double output;
	double delta;
} neuron_t;

typedef struct
{
	unsigned int num_neurons;
	neuron_t* neurons;
} layer_t;

typedef struct
{
	unsigned int num_inputs;
	unsigned int num_layers;
	layer_t* layers;
	ActivateFunc activation;
	ActivateDerivative activation_derivative;
} network_t;

// random weight init
double random_weight()
{
	return (double)rand() / RAND_MAX * 2 - 1; // decently small
}

// init neuron
neuron_t create_neuron(unsigned int num_inputs)
{
	neuron_t neuron = { 0 };
	neuron.weights = (double*)malloc(num_inputs * sizeof(double));
	for (unsigned int i = 0; i < num_inputs; ++i)
	{
		neuron.weights[i] = random_weight();
	}

	neuron.bias = random_weight();
	neuron.output = 0;
	neuron.delta = 0;
	return neuron;
}

// init layer
layer_t create_layer(unsigned int num_neurons, unsigned int num_inputs_per_neuron)
{
	layer_t layer = { 0 };
	layer.num_neurons = num_neurons;
	layer.neurons = (neuron_t*)malloc(num_neurons * sizeof(neuron_t));
	for (unsigned int i = 0; i < num_neurons; ++i)
	{
		layer.neurons[i] = create_neuron(num_inputs_per_neuron);
	}
	return layer;
}

// init network
network_t create_network(unsigned int num_inputs, int* neurons_per_layer, unsigned int num_layers, ActivateFunc act_func, ActivateDerivative act_deriv)
{
	network_t network = { 0 };
	network.num_inputs = num_inputs;
	network.num_layers = num_layers;
	network.activation = act_func;
	network.activation_derivative = act_deriv;

	network.layers = (layer_t*)malloc(num_layers * sizeof(layer_t));

	// Input is connected to the first hidden layer
	network.layers[0] = create_layer(neurons_per_layer[0], num_inputs);

	// Hidden & output layers
	for (unsigned int i = 1; i < num_layers; ++i)
	{
		network.layers[i] = create_layer(neurons_per_layer[i], neurons_per_layer[i - 1]);
	}

	return network;
}

// forward func
void network_forward(network_t* network, double* input)
{
	for (unsigned int i = 0; i < network->layers[0].num_neurons; ++i)
	{
		double sum = 0;
		for (unsigned int j = 0; j < network->num_inputs; ++j)
		{
			sum += input[j] * network->layers[0].neurons[i].weights[j];
		}
		sum += network->layers[0].neurons[i].bias;
		network->layers[0].neurons[i].output = network->activation(sum);
	}

	for (unsigned int l = 1; l < network->num_layers; ++l)
	{
		for (unsigned int i = 0; i < network->layers[l].num_neurons; ++i)
		{
			double sum = 0;
			for (unsigned int j = 0; j < network->layers[l - 1].num_neurons; ++j)
			{
				sum += network->layers[l - 1].neurons[j].output * network->layers[l].neurons[i].weights[j];
			}
			sum += network->layers[l].neurons[i].bias;
			network->layers[l].neurons[i].output = network->activation(sum);
		}
	}
}

// backpropagation function
void backpropagation(network_t* network, double* inputs, double* target, double learning_rate)
{
	int output_layer = network->num_layers - 1;

	// output layer deltas
	for (unsigned int i = 0; i < network->layers[output_layer].num_neurons; ++i)
	{
		double output = network->layers[output_layer].neurons[i].output;
		double error = target[i] - output;
		network->layers[output_layer].neurons[i].delta = error * network->activation_derivative(output);
	}

	// hidden layers deltas
	for (int l = output_layer - 1; l >= 0; --l)
	{
		for (unsigned int i = 0; i < network->layers[l].num_neurons; ++i)
		{
			double output = network->layers[l].neurons[i].output;
			double error = 0;
			for (unsigned int j = 0; j < network->layers[l + 1].num_neurons; ++j)
			{
				error += network->layers[l + 1].neurons[j].delta * network->layers[l + 1].neurons[j].weights[i];
			}
			network->layers[l].neurons[i].delta = error * network->activation_derivative(output);
		}
	}

	// Update weights for the output layer
	for (unsigned int i = 0; i < network->layers[output_layer].num_neurons; ++i)
	{
		for (unsigned int j = 0; j < network->layers[output_layer - 1].num_neurons; ++j)
		{
			network->layers[output_layer].neurons[i].weights[j] += learning_rate * network->layers[output_layer].neurons[i].delta * network->layers[output_layer - 1].neurons[j].output;
		}
		network->layers[output_layer].neurons[i].bias += learning_rate * network->layers[output_layer].neurons[i].delta;
	}

	// Update weights for the hidden layers
	for (int l = output_layer - 1; l >= 0; --l)
	{
		for (unsigned int i = 0; i < network->layers[l].num_neurons; ++i)
		{
			for (unsigned int j = 0; j < (l == 0 ? network->num_inputs : network->layers[l - 1].num_neurons); ++j)
			{
				network->layers[l].neurons[i].weights[j] += learning_rate * network->layers[l].neurons[i].delta * (l == 0 ? inputs[j] : network->layers[l - 1].neurons[j].output);
			}
			network->layers[output_layer].neurons[i].bias += learning_rate * network->layers[output_layer].neurons[i].delta;
		}
	}
}

// train the network
void train(network_t* network, double** inputs, double** targets, int data_size, int epochs, double learning_rate)
{
	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		for (int i = 0; i < data_size; ++i)
		{
			network_forward(network, inputs[i]);
			backpropagation(network, inputs[i], targets[i], learning_rate);
		}
	}
}

// test the network
void test(network_t* network, double* inputs)
{
	network_forward(network, inputs);
	for (unsigned int i = 0; i < network->layers[network->num_layers - 1].num_neurons; ++i)
	{
		printf("Output: %lf\n", network->layers[network->num_layers - 1].neurons[i].output);
	}

	// 
}

int main(int argc, char* argv[])
{
	srand((unsigned int)time(NULL));

	// XOR problem
	double _inputs[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	double _outputs[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

	// network config
	int layers[] = { 2, 1 }; // hidden layer with 2 neurons, output with 1 neuron
	network_t network = create_network(2, layers, 2, sigmoid, sigmoid_derivative);

	// train the network
	double* inputs[] = { _inputs[0], _inputs[1], _inputs[2], _inputs[3] };
	double* targets[] = { _outputs[0], _outputs[1], _outputs[2], _outputs[3] };
	train(&network, inputs, targets, 4, 10000, 0.5);

	// test
	double test_input[2] = { 0, 1 };
	test(&network, test_input);
	
	for (unsigned int i = 0; i < network.num_layers; ++i)
	{
		layer_t* crnt = network->layers[i];
		for (unsigned int j = 0; j < crnt->num_neurons; ++j)
		{
			neuron_t* neuron = crnt->neurons[j];
			free(neuron->weights);
		}
		free(crnt->neurons);
	}
	
	free(network->layers);

	return 0;
}
