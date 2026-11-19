import math
import random


class LinearRegression:
    def __init__(self):
        self.gradients = []
        self.intercept = 0
        self.data = []
        self.target = []
        self.iterations = 1_000_000
        self.steps = 0.0001
        self.dimens = 1
        self.temp = 0.1

    def allow(self, delta_loss):
        x = math.exp(
            -delta_loss / self.temp
        )  # This is the Boltzmann probability function: \(P = \exp(-\Delta \text{Loss} / T)\).
        if random.random() < x:
            return True

    def pred(self, x: list[float], m: list[float] = None, b: float = None):
        sumn = 0
        for i in range(self.dimens):
            if m is None or b is None:
                m = self.gradients
                b = self.intercept
            t = m[i] * x[i]
            sumn += t
        if b is None:
            b = self.intercept
        return sumn + b

    def error_rate(
        self,
        data: list[list[float]],
        target: list[float],
        m: list[float] = None,
        b: float = None,
    ):
        if m is None or b is None:
            m = self.gradients
            b = self.intercept
        sumn = 0
        for k in range(len(data)):
            sumn += (target[k] - self.pred(data[k], m, b)) ** 2
        sumn = sumn / len(data)
        return sumn

    def choose(self, x: int, step):
        liss = []
        for i in range(x + 1):
            y = [0] * (x + 1)
            y[i] = step
            liss.append(tuple(y))
            y[i] = -step
            liss.append(tuple(y))
        return liss

    def fit(self, data: list, targets: list[float]):
        if type(data[0]) != list:
            data = [[x] for x in data]
        self.dimens = len(data[0])
        self.data = data
        self.target = targets
        self.gradients = [0] * self.dimens
        steps = self.steps
        iters = self.iterations
        x = []
        nm = [0] * self.dimens

        current_best_error = self.error_rate(
            data, targets, self.gradients, self.intercept
        )
        for j in range(iters):
            for m in range(self.dimens):
                for choose in self.choose(self.dimens, steps):
                    for p in range(self.dimens):
                        nm[p] = self.gradients[p] + choose[p]
                    nb = self.intercept + choose[-1]
                    e = self.error_rate(data, targets, nm, nb)
                    if e < current_best_error:
                        self.intercept = nb
                        self.gradients = nm
                        nm = self.gradients.copy()
                        current_best_error = self.error_rate(data, targets, nm, nb)
                    elif self.allow(e - current_best_error):
                        self.intercept = nb
                        self.gradients = nm
                        nm = self.gradients.copy()
                        current_best_error = self.error_rate(data, targets, nm, nb)
                    else:
                        pass
            if j % 10 == 0:
                self.temp *= 0.999
            if j % 100_000 == 0:

                if x == [round(self.pred(i), 3) for i in self.data]:
                    break
                x = [round(self.pred(i), 3) for i in self.data]
                print(f"Epoch {j}:", x)
            self.steps *= 0.999


data_test = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [16, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]
target_test = [((8 * num[0]) + (1 * num[1])) + 3 for num in data_test]
print("Target is", target_test)
model = LinearRegression()
model.fit(data_test, target_test)
print(
    f"Possible gradients are {model.gradients}, y-intercept is {round(model.intercept,2)}"
)
