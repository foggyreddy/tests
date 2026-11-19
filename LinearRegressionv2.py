import math
import random


class LinearRegression:
    def __init__(self):
        self.gradient = 0
        self.intercept = 0
        self.data = []
        self.target = []
        self.temp = 0.3

    def allow(self, delta_loss):
        x = math.exp(
            -delta_loss / self.temp
        )  # This is the Boltzmann probability function: \(P = \exp(-\Delta \text{Loss} / T)\).
        if random.random() < x:
            return True

    def pred(self, x: float, m: float = None, b: float = None):
        if m is None or b is None:
            m = self.gradient
            b = self.intercept
        return (m * x) + b

    def error_rate(
        self, data: list[float], target: list[float], m: float = None, b: float = None
    ):
        if m is None or b is None:
            m = self.gradient
            b = self.intercept
        sumn = 0
        for k in range(len(data)):
            sumn += (target[k] - self.pred(data[k], m, b)) ** 2
        sumn = sumn / len(data)
        return sumn

    def fit(self, data: list[float], targets: list[float]):
        self.data = data
        self.target = targets
        self.gradient = 0
        self.intercept = sum(data) / len(data)
        steps = 0.0001
        iters = 1_000_000
        current_best_error = self.error_rate(
            data, targets, self.gradient, self.intercept
        )
        for j in range(iters):
            for dm, db in [(steps, 0), (-steps, 0), (0, steps), (0, -steps)]:
                nm = self.gradient + dm
                nb = self.intercept + db
                e = self.error_rate(data, targets, nm, nb)
                if e < current_best_error:
                    self.intercept = nb
                    self.gradient = nm
                    current_best_error = self.error_rate(data, targets, nm, nb)
                elif self.allow(e - current_best_error):
                    self.intercept = nb
                    self.gradient = nm
                    current_best_error = self.error_rate(data, targets, nm, nb)
                else:
                    pass
            self.temp *= 0.999
            if j % 100_000 == 0:
                x = [round(model.pred(i), 2) for i in self.data]
                print(f"Epoch {j}:", x)


data_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
target_test = [(3 * num) + 4 for num in data_test]
print("Target is", target_test)
model = LinearRegression()
model.fit(data_test, target_test)
print(
    f"Gradient is {round(model.gradient,2)}, y-intercept is {round(model.intercept,2)}"
)
