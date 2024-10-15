import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        return input_grad
    
    def update(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


class Activation1(_Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, output_grad):
        return output_grad * np.where(self.input > 0, 1, self.alpha)

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        '''Softmax'''
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.predict = exp_input / np.sum(exp_input, axis=1, keepdims=True)

        '''Average loss'''
        self.target = target.astype(int)
        your_loss = -np.mean(np.log(self.predict[range(self.target.shape[0]), self.target]))

        return self.predict, your_loss

    def backward(self):
        input_grad = self.predict.copy()
        input_grad[range(self.target.shape[0]), self.target] -= 1
        input_grad /= self.target.shape[0]
    
        return input_grad
    

class ConvLayer(_Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros((out_channels, 1))
        
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        batch_size, _, input_height, input_width = input.shape
        
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        self.output = np.zeros((batch_size, self.out_channels, output_height, output_width))
        
        if self.padding > 0:
            padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_input = input
        
        for i in range(output_height):
            for j in range(output_width):
                input_slice = padded_input[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    self.output[:, k, i, j] = np.sum(input_slice * self.weights[k], axis=(1, 2, 3)) + self.biases[k]
        
        return self.output

    def backward(self, output_grad):
        batch_size, _, output_height, output_width = output_grad.shape
        input_grad = np.zeros_like(self.input)
        weights_grad = np.zeros_like(self.weights)
        biases_grad = np.zeros_like(self.biases)

        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            padded_input_grad = np.pad(input_grad, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_input = self.input
            padded_input_grad = input_grad

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                input_slice = padded_input[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    weights_grad[k] += np.tensordot(input_slice, output_grad[:, k, i, j], axes=((0), (0)))
                    padded_input_grad[:, :, h_start:h_end, w_start:w_end] += self.weights[k] * output_grad[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis]
                
                biases_grad[:, 0] += np.sum(output_grad[:, :, i, j], axis=0)

        if self.padding > 0:
            input_grad = padded_input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = padded_input_grad

        self.weights_grad = weights_grad
        self.biases_grad = biases_grad

        return input_grad

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad

    
class BatchNormalization(_Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        self.input = None
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.input_shape = None

    def forward(self, input, training=True):
        self.input_shape = input.shape
        
        if len(input.shape) == 2:
            input = input.reshape(input.shape[0], self.num_features, 1, 1)
        
        if training:
            batch_mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(input, axis=(0, 2, 3), keepdims=True)
            
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            
            normalized = (input - batch_mean) / np.sqrt(batch_var + self.eps)
            output = self.gamma * normalized + self.beta
            
            self.input = input
            self.normalized = normalized
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
            output = self.gamma * normalized + self.beta
        
        if len(self.input_shape) == 2:
            output = output.reshape(self.input_shape)
        
        return output

    def backward(self, output_grad):
        if len(output_grad.shape) == 2:
            output_grad = output_grad.reshape(output_grad.shape[0], self.num_features, 1, 1)
        
        self.gamma_grad = np.sum(output_grad * self.normalized, axis=(0, 2, 3), keepdims=True)
        self.beta_grad = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)
        
        normalized_grad = output_grad * self.gamma
        
        var_grad = np.sum(normalized_grad * (self.input - self.batch_mean), axis=(0, 2, 3), keepdims=True) * -0.5 * np.power(self.batch_var + self.eps, -1.5)
        
        mean_grad = np.sum(normalized_grad, axis=(0, 2, 3), keepdims=True) * -1 / np.sqrt(self.batch_var + self.eps) + var_grad * np.mean(-2 * (self.input - self.batch_mean), axis=(0, 2, 3), keepdims=True)
        
        N = self.input.shape[0] * self.input.shape[2] * self.input.shape[3]
        input_grad = normalized_grad / np.sqrt(self.batch_var + self.eps) + var_grad * 2 * (self.input - self.batch_mean) / N + mean_grad / N
        
        if len(self.input_shape) == 2:
            input_grad = input_grad.reshape(self.input_shape)
        
        return input_grad

    def update(self, learning_rate):
        self.gamma -= learning_rate * self.gamma_grad
        self.beta -= learning_rate * self.beta_grad

class MaxPooling(_Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.max_indices = None

    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape
        pool_height = (height - self.pool_size) // self.stride + 1
        pool_width = (width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, pool_height, pool_width))
        self.max_indices = np.zeros(self.output.shape, dtype=int)

        for i in range(pool_height):
            for j in range(pool_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                pool_region = input[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.max(pool_region, axis=(2, 3))
                self.max_indices[:, :, i, j] = np.argmax(pool_region.reshape(batch_size, channels, -1), axis=2)

        return self.output

    def backward(self, output_grad):
        batch_size, channels, height, width = self.input.shape
        input_grad = np.zeros_like(self.input)

        for i in range(output_grad.shape[2]):
            for j in range(output_grad.shape[3]):
                h_start = i * self.stride
                h_end = min(h_start + self.pool_size, height)
                w_start = j * self.stride
                w_end = min(w_start + self.pool_size, width)

                for b in range(batch_size):
                    for c in range(channels):
                        max_idx = self.max_indices[b, c, i, j]

                        h_idx = h_start + max_idx // (w_end - w_start)
                        w_idx = w_start + max_idx % (w_end - w_start)

                        if h_idx < height and w_idx < width:
                            input_grad[b, c, h_idx, w_idx] += output_grad[b, c, i, j]

        return input_grad