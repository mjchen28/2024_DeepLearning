from .layer import *

from .layer import *

class Network(object):
    def __init__(self):
        # Define layers
        self.conv1 = ConvLayer(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNormalization(16)
        self.act1 = Activation1()
        self.pool1 = MaxPooling(2, 2)

        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNormalization(32)
        self.act2 = Activation1()
        self.pool2 = MaxPooling(2, 2)

        self.fc1 = FullyConnected(7 * 7 * 32, 128)
        self.bn3 = BatchNormalization(128)
        self.act3 = Activation1()

        self.fc2 = FullyConnected(128, 10)
        self.loss = SoftmaxWithloss()

    def forward(self, input, target, training=True):
        input = input.reshape(-1, 1, 28, 28)

        h1 = self.conv1.forward(input)
        h1_bn = self.bn1.forward(h1, training)
        h1_act = self.act1.forward(h1_bn)
        h1_pool = self.pool1.forward(h1_act)

        h2 = self.conv2.forward(h1_pool)
        h2_bn = self.bn2.forward(h2, training)
        h2_act = self.act2.forward(h2_bn)
        h2_pool = self.pool2.forward(h2_act)

        h2_flat = h2_pool.reshape(h2_pool.shape[0], -1)

        h3 = self.fc1.forward(h2_flat)
        h3_bn = self.bn3.forward(h3, training)
        h3_act = self.act3.forward(h3_bn)

        h4 = self.fc2.forward(h3_act)

        pred, loss = self.loss.forward(h4, target)
        return pred, loss

    def backward(self):
        loss_grad = self.loss.backward()

        h4_grad = self.fc2.backward(loss_grad)
        h3_act_grad = self.act3.backward(h4_grad)
        h3_bn_grad = self.bn3.backward(h3_act_grad)
        h3_grad = self.fc1.backward(h3_bn_grad)

        h2_pool_grad = h3_grad.reshape(-1, 32, 7, 7)

        h2_act_grad = self.pool2.backward(h2_pool_grad)
        h2_bn_grad = self.act2.backward(h2_act_grad)
        h2_grad = self.bn2.backward(h2_bn_grad)
        h1_pool_grad = self.conv2.backward(h2_grad)

        h1_act_grad = self.pool1.backward(h1_pool_grad)
        h1_bn_grad = self.act1.backward(h1_act_grad)
        h1_grad = self.bn1.backward(h1_bn_grad)
        input_grad = self.conv1.backward(h1_grad)

        return input_grad

    def update(self, lr, optimizer=None):
        if optimizer != None:
            optimizer.update(self.conv1, lr)
            optimizer.update(self.bn1, lr)
            optimizer.update(self.conv2, lr)
            optimizer.update(self.bn2, lr)
            optimizer.update(self.fc1, lr)
            optimizer.update(self.bn3, lr)
            optimizer.update(self.fc2, lr)
        else:
            self.conv1.update(lr)
            self.bn1.update(lr)
            self.conv2.update(lr)
            self.bn2.update(lr)
            self.fc1.update(lr)
            self.bn3.update(lr)
            self.fc2.update(lr)

class CosineAnnealingScheduler:
    def __init__(self, lr_0, T_max, eta_min=0):
        """
            lr_0 (float): Initial learning rate.
            T_max (int): Maximum number of iterations/epochs for one cycle.
            eta_min (float): Minimum learning rate after annealing. Default is 0.
        """
        self.lr_0 = lr_0
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_epoch = 0

    def get_lr(self):
        """
        Calculate the current learning rate using the cosine annealing formula.
        """
        cos_inner = np.pi * self.current_epoch / self.T_max
        lr_t = self.eta_min + 0.5 * (self.lr_0 - self.eta_min) * (1 + np.cos(cos_inner))
        return lr_t

    def step(self):
        """
        Move to the next epoch/iteration.
        """
        self.current_epoch += 1