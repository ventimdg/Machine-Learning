from .utils import LayerTest
from neural_networks.activations import Linear, Sigmoid, TanH, ReLU, SoftMax, ArcTan

class TestLinear(LayerTest):

    LayerCls  = Linear

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()


class TestSigmoid(LayerTest):

    LayerCls  = Sigmoid

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()


class TestTanH(LayerTest):

    LayerCls  = TanH
    BatchSizes = (64,)
    InputSizes = ((31,),)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()


class TestReLU(LayerTest):

    LayerCls  = ReLU

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()

class TestSoftMax(LayerTest):

    LayerCls  = SoftMax

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()

class TestArcTan(LayerTest):

    LayerCls  = ArcTan
    BatchSizes = (64,)
    InputSizes = ((31,),)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

    # def test_check_grad(self):
    #     return self._check_gradients()

