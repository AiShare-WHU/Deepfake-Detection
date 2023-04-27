import numpy as np
from math import factorial

class BIW():
    def __init__(self, ff, rf, fr, rr, max_length=20):
        self.ff = ff  # a
        self.rf = rf  # b
        self.fr = fr  # c
        self.rr = rr  # d
        self.max_length = max_length
        self.fake_table, self.real_table = self.build_table()

    def cal_reality(self, k, fake=True):
        if fake:
            p = sum(self.comb(i,k)*np.power(self.ff/(self.ff+self.rf),i)*np.power(self.rf/(self.ff+self.rf), k-i)
                    for i in range(np.ceil(k / 2).astype(np.int16), k+1))
        else:
            p = sum(self.comb(i,k)*np.power(self.rr/(self.rr+self.fr),i)*np.power(self.fr/(self.rr+self.fr), k-i)
                    for i in range(np.ceil(k / 2).astype(np.int16), k+1))
        return p

    def comb(self, k, n):
        return factorial(n) / factorial(k) / factorial(n - k)

    def shift_weight(self, reality, mode="line"):
        if mode == "add":
            weights = np.ones_like(reality)
            weights = weights + (reality - np.mean(reality))
            weights = weights / len(weights)
        elif mode == "line":
            weights = reality / np.mean(reality) / len(reality)
        elif mode == "softmax":
            weights = self.softmax(reality)
        else:
            raise KeyError("no mode named %s" % mode)
        return weights

    def softmax(self, logits):
        e_x = np.exp(logits)
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return probs


    def check_len(self, result, threshold=0.5):
        lens = []
        flag = True
        count = 0
        for r in result:
            if r >= threshold and flag:
                count += 1
            if r >= threshold and not flag:
                lens.append(count)
                count = 1
                flag = True
            if r < threshold and not flag:
                count += 1
            if r < threshold and flag:
                lens.append(count)
                count = 1
                flag = False
        lens.append(count)
        app_len = []
        for l in lens:
            app_len.extend([l] * l)
        return app_len

    def build_table(self):
        fake_table = np.zeros(self.max_length+1)
        for i in range(1, self.max_length+1):
            fake_table[i] = self.cal_reality(i)
        real_table = np.zeros(self.max_length+1)
        for i in range(1, self.max_length+1):
            real_table[i] = self.cal_reality(i, fake=False)

        return fake_table, real_table

    def cal_result(self, results, mode="add"):
        lens = self.check_len(results)
        realities = np.zeros_like(results)
        p = 0
        for i, r in enumerate(results):
            if r >= 0.5:
                realities[i] = self.fake_table[lens[i]]
            else:
                realities[i] = self.real_table[lens[i]]
        weights = self.shift_weight(realities, mode=mode)
        return np.dot(results, weights)

if __name__ == "__main__":
    biw = BIW(400, 0, 0, 450)
    r = np.array([0.7,0.7,0.1,0.6,0.1,0.7,0.7,0.7,0.1])
    print(biw.cal_result(r, mode="add"), r.mean())

