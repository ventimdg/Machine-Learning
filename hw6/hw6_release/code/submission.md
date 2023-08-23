### Activation Function Implementations:

Implementation of `activations.Linear`:

```python
class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        return dY

```

Implementation of `activations.Sigmoid`:

```python
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        return ...

```

Implementation of `activations.ReLU`:

```python
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return np.maximum(0, Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        dY[Z < 0] = 0
        return dY

```

Implementation of `activations.SoftMax`:

```python
class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###

        maxes = np.array(np.max(Z, axis=1))
        maxes = np.reshape(maxes, (np.shape(maxes)[0], 1))
        
        pre_exponential =  Z - maxes
        post_exponential = np.exp(pre_exponential)

        sums = np.sum(post_exponential, axis=1)
        sums = np.reshape(sums, (np.shape(sums)[0], 1))

        answer = np.divide(post_exponential, sums)

        return answer

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        forward = self.forward(Z)
        answer = np.zeros_like(Z)
        for i in range(np.shape(Z)[0]):
            data =  forward[i]
            outer = np.multiply(-1, np.outer(data, data))
            diagonal = np.diag(outer)
            diagonal= np.multiply(-1, diagonal)
            diagonal = np.sqrt(diagonal)
            diagonal = np.diag(diagonal)
            jacobian = diagonal + outer
            answer[i] = dY[i] @ jacobian
        return answer

```


### Layer Implementations:

Implementation of `layers.FullyConnected`:

```python
class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict()  # cache for backprop
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b

        self.cache["X"] = X
        self.cache["Z"] = Z
        
        # perform an affine transformation and activation
        out = self.activation(Z)
        
        # store information necessary for backprop in `self.cache`

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X = self.cache["X"] 
        Z = self.cache["Z"]

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        W = self.parameters["W"]
        
        dLdZ = self.activation.backward(Z, dLdY)
        dLdX = dLdZ @ W.T
        dLdW = X.T @ dLdZ
        dLdb = np.ones((1, np.shape(dLdZ)[0])) @ dLdZ


        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.


        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdb

        ### END YOUR CODE ###

        return dLdX

```

Implementation of `layers.Pool2D`:

```python
class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass

        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_shape[0], self.kernel_shape[1]


        ### BEGIN YOUR CODE ###

        # implement a convolutional forward pass

        padded = np.pad(X, ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode= 'constant')

        padded_rows  = in_rows + 2*self.pad[0]
        padded_cols = in_cols + 2*self.pad[1]
        filtered_rows = padded_rows - kernel_height
        filtered_cols = padded_cols - kernel_width

        num_output_rows = int(filtered_rows / self.stride + 1) 
        num_output_cols  = int(filtered_cols / self.stride + 1)

        
        answer = np.zeros((n_examples, num_output_rows, num_output_cols, in_channels)) 
        
        for row in range(num_output_rows):
            
            for col in range(num_output_cols):

                padded_slice = padded[:, row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :]

                answer[:, row, col, :] = self.pool_fn(padded_slice,  axis=(1, 2))
        
        
        self.cache["out_rows"] = num_output_rows
        self.cache["out_cols"] = num_output_cols
        self.cache["in_rows"] = in_rows
        self.cache["in_cols"] = in_cols
        self.cache["X_pad"] = padded

        # cache any values required for backprop

        ### END YOUR CODE ###

        return answer

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###
    

        padded = self.cache["X_pad"]
        num_output_rows = self.cache["out_rows"]
        num_output_cols = self.cache["out_cols"]
        in_rows = self.cache["in_rows"] 
        in_cols = self.cache["in_cols"] 
        kernel_height, kernel_width = self.kernel_shape[0], self.kernel_shape[1]
        average_div = kernel_height * kernel_width
        padded_out = np.zeros_like(padded)


        for row in range(num_output_rows):
            for col in range(num_output_cols):

                if self.mode == "average":
                    
                    derivative =  dLdY[:, row : row + 1, col : col + 1, :] / average_div

                    padded_out[:, row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :] += derivative

                if self.mode == "max":

                    padded_slice = padded[:, row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :]

                    flattened_spatial = padded_slice.reshape(padded_slice.shape[0], -1, padded_slice.shape[3]) 

                    removed = (flattened_spatial == np.max(flattened_spatial, axis=1, keepdims=True))

                    removed = removed.reshape(padded_slice.shape[0], kernel_height, kernel_width, padded_slice.shape[3])

                    padded_out[:, row * self.stride : row * self.stride + kernel_height , col * self.stride : col * self.stride + kernel_width, :] += dLdY[:, row:row+1, col:col+1, :] * removed
        
        return padded_out[:, self.pad[0]:in_rows+self.pad[0], self.pad[1]:in_cols+self.pad[1], :] 

```

Implementation of `layers.Conv2D.__init__`:

```python
    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

```

Implementation of `layers.Conv2D._init_parameters`:

```python
    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

```

Implementation of `layers.Conv2D.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)


        ### BEGIN YOUR CODE ###

        # implement a convolutional forward pass

        padded = np.pad(X, ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode= 'constant')

        padded_rows  = in_rows + 2*self.pad[0]
        padded_cols = in_cols + 2*self.pad[1]
        filtered_rows = padded_rows - kernel_height
        filtered_cols = padded_cols - kernel_width

        num_output_rows = int(filtered_rows / self.stride + 1) 
        num_output_cols  = int(filtered_cols / self.stride + 1)

        
        Z = np.zeros((n_examples, num_output_rows, num_output_cols, out_channels)) 
        
        for row in range(num_output_rows):
            for col in range(num_output_cols):
                for channel in range(out_channels):
                    padded_slice = padded[:, row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :]
                    weight_slice = W[:, :, :, channel]
                    convolved = padded_slice * weight_slice
                    Z[:, row, col, channel] = np.einsum('ijkl->i', convolved) + b[:, channel]
        
        self.cache["Z"] = Z 
        self.cache["X"] = X

        # cache any values required for backprop

        ### END YOUR CODE ###


        return self.activation(Z)

```

Implementation of `layers.Conv2D.backward`:

```python
    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###

        X = self.cache["X"]
        Z = self.cache["Z"] 

        W = self.parameters["W"] 
        
        kernel_height, kernel_width, in_channels, out_channels = W.shape 
        n_examples, in_rows, in_cols, in_channels = X.shape

        padded = np.pad(X, ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode= 'constant')

        padded_rows  = in_rows + 2*self.pad[0]
        padded_cols = in_cols + 2*self.pad[1]
        filtered_rows = padded_rows - kernel_height
        filtered_cols = padded_cols - kernel_width

        num_output_rows = int(filtered_rows / self.stride + 1) 
        num_output_cols  = int(filtered_cols / self.stride + 1)

        dLdZ = self.activation.backward(Z, dLdY)

        padded_out = np.zeros_like(padded)
        dLdW = np.zeros_like(W)

        self.gradients["b"] = np.einsum('ijkl->l', dLdZ)

        for row in range(num_output_rows):
            for col in range(num_output_cols):
                for channel in range(out_channels):

                    padded_slice = padded[:, row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :]
                    weight_slice = W[None, :, :, :, channel]
                    dldz_slice =  dLdZ[:, row : row + 1, col : col + 1, None, channel]

                    padded_out[:,row * self.stride : row * self.stride + kernel_height, col * self.stride : col * self.stride + kernel_width, :] += weight_slice * dldz_slice

                    convolved  = padded_slice * dldz_slice

                    dLdW[:, :, :, channel] += np.einsum("ijkl->jkl", convolved)
        
        self.gradients["W"] = dLdW 

        return padded_out[:, self.pad[0]:in_rows+self.pad[0], self.pad[1]:in_cols+self.pad[1], :] 

```


### Loss Function Implementations:

Implementation of `losses.CrossEntropy`:

```python
class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """

        ### YOUR CODE HERE ###
        losses = Y * np.log(Y_hat)
        sum_losses = -np.sum(losses)
        answer = sum_losses / np.shape(Y)[0]
        return answer

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the derivative of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        ### YOUR CODE HERE ###
        vector = np.multiply(-1, np.divide(Y, Y_hat))
        answer = np.divide(vector, np.shape(Y)[0])
        return answer

```

Implementation of `losses.L2`:

```python
class L2(Loss):
    """Mean squared error loss."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Compute the mean squared error loss for predictions `Y_hat` given
        regression targets `Y`.

        Parameters
        ----------
        Y      vector of regression targets of shape (batch_size, 1)
        Y_hat  vector of predictions of shape (batch_size, 1)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass for mean squared error loss.

        Parameters
        ----------
        Y      vector of regression targets of shape (batch_size, 1)
        Y_hat  vector of predictions of shape (batch_size, 1)

        Returns
        -------
        the derivative of the mean squared error with respect to the last layer
        of the neural network
        """
        ### YOUR CODE HERE ###
        return ...

```


### Model Implementations:

Implementation of `models.NeuralNetwork.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through all the layers of the neural network.

        Parameters
        ----------
        X  design matrix whose must match the input shape required by the
           first layer

        Returns
        -------
        forward pass output, matches the shape of the output of the last layer
        """
        ### YOUR CODE HERE ###
        # Iterate through the network's layers.
        for layer in self.layers:
            X = layer.forward(X)
        return X

```

Implementation of `models.NeuralNetwork.backward`:

```python
    def backward(self, target: np.ndarray, out: np.ndarray) -> float:
        """One backward pass through all the layers of the neural network.
        During this phase we calculate the gradients of the loss with respect to
        each of the parameters of the entire neural network. Most of the heavy
        lifting is done by the `backward` methods of the layers, so this method
        should be relatively simple. Also make sure to compute the loss in this
        method and NOT in `self.forward`.

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on training data

        Returns
        -------
        the loss of the model given the training inputs and targets
        """
        ### YOUR CODE HERE ###
        # Compute the loss.
        # Backpropagate through the network's layers.
        curr_loss = self.loss.forward(target, out)
        derivative = self.loss.backward(target, out)
        backwards = self.layers[::-1]
        for layer in backwards:
            derivative = layer.backward(derivative)
        return curr_loss

```

Implementation of `models.NeuralNetwork.predict`:

```python
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a forward and backward pass to calculate the predictions and
        loss of the neural network on the given data.

        Parameters
        ----------
        X  input features
        Y  targets (same length as `X`)

        Returns
        -------
        a tuple of the prediction and loss
        """
        ### YOUR CODE HERE ###
        # Do a forward pass. Maybe use a function you already wrote?
        # Get the loss. Remember that the `backward` function returns the loss.
        answer = self.forward(X)
        loss = self.backward(Y, answer)
        return (answer, loss)

```

