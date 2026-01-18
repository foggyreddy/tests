class LinearRegression:
    def __init__(self):
        self.gradient = 0
        self.intercept = 0
    def fit(self, data: list[float], target: list[float]):
        steps = (max(data) - min(data)+1)/10
        bot = min(data)
        xs = []
        ys = []
        for i in range(10):
            sumn = 0
            nom = 0
            xs.append(bot + (steps/2))
            for i, datum in enumerate(data):
                if bot <= datum <= bot+steps:
                    sumn += target[i]
                    nom += 1
            ys.append(sumn/nom)
            bot += steps
        print(xs, ys)
        self.intercept = ys[0] - self.gradient * xs[0]
        self.steps = steps


class LogisticalRegression
