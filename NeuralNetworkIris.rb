require "matrix"
string = File.read("iris.data").strip.split("\n")

output = []
data = string.map do |line|
  line = line.split(",")
  output << case (line.last)
  when /Setosa/i then [1, 0, 0]
  when /Versicolor/i then [0, 1, 0]
  else [0, 0, 1]
  end
  line[0..3].map { |e| e.to_f }
end

input_matrix = Matrix[*data]
output_matrix = Matrix[*output]

def activation_function(x)
  Math.tanh(x)
end

def initial_weights(rows, cols)
  weights = 1.upto(rows).map do
    1.upto(cols).map do
      rand / 10 - 0.05
    end
  end
  Matrix[*weights]
end

input_weights = initial_weights(5, 8)
hidden_weights = initial_weights(9, 3)

def calculate_neuron_activation(input, weights)
  input_with_bias = input.to_a.map { |x| [1] + x }
  input_with_bias = Matrix[*input_with_bias]
  output = input_with_bias * weights
  output.map { |element| activation_function(element) }
end

def predict(input, input_weights, hidden_weights)
  hidden_layer_input = calculate_neuron_activation(input, input_weights)
  calculate_neuron_activation(hidden_layer_input, hidden_weights)
end

#Lembrando: output_matrix é a matriz que preparamos, com os dados já classificados
def calculate_gradient(input, input_weights, hidden_weights, output_matrix)
  hidden_layer_input = calculate_neuron_activation(input, input_weights)
  output = calculate_neuron_activation(hidden_layer_input, hidden_weights)

  #Custo desta solução:
  size = output_matrix.row_count
  error = output - output_matrix
  squared_error = error.map { |e| e * e }
  mean_squared_error = squared_error.to_a.flatten.inject(0) { |r, e| r + e } / (2 * row_count)

  #Cálculo dos gradientes
  hidden_layer_input_with_bias = hidden_layer_input.to_a.map { |x| [1] + x }
  hidden_layer_input_with_bias = Matrix[*hidden_layer_input_with_bias]
  gradient1 = (error.transpose * hidden_layer_input_with_bias).transpose / size
end
