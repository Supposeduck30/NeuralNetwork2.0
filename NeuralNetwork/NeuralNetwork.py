import math
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------ elementary operations ------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    # ------------ activations ------------
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # ------------ autodiff kick-off ------------
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

# ---------------- Network components ----------------
class Neuron:
    def __init__(self, nin):
        # Xavier/Glorot uniform in [-1/√nin, +1/√nin]
        bound = 1 / math.sqrt(nin)
        self.w = [Value(random.uniform(-bound, bound)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x           # final output still tanh

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# ---------------- Training ----------------
random.seed(0)                       # reproducibility
model = MLP(2, [4, 4, 1])            # 2-4-4-1 MLP

data = [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]

for epoch in range(4000):
    total_loss = 0.0
    for x, y in data:
        x_val = [Value(v) for v in x]
        y_pred = model(x_val)[0]
        loss = (y_pred - y) ** 2      # MSE
        total_loss += loss.data

        # backprop
        for p in model.parameters():
            p.grad = 0
        loss.backward()

        # SGD update
        for p in model.parameters():
            p.data -= 0.1 * p.grad

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {total_loss:.4f}")

print("\nFinal predictions:")
for x, y in data:
    x_val = [Value(v) for v in x]
    pred = model(x_val)[0].data
    print(f"Input {x} → Predicted {pred:.4f}  (actual {y})")
