import numpy as np
import math
import random
from time import time_ns

class Neurosa:
    def __init__(self, Q, max_iter=1e8, thld_max=8e4, thld_delta=2e-3):
        self.Q = Q
        self.N, self.M = Q.shape
        self.MAX_ITER = max_iter
        self.thld_max = thld_max
        self.thld_delta = thld_delta
        self.DTYPE = np.float32
        
        self.ds_p = np.zeros((self.N,), dtype=self.DTYPE)
        self.ds_n = np.zeros_like(self.ds_p, dtype=self.DTYPE)
        self.s_p = np.ones_like(self.ds_p, dtype=self.DTYPE)
        self.s_n = np.zeros_like(self.ds_p, dtype=self.DTYPE)
        self.vmem = np.matmul(Q, (self.s_p - self.s_n))
        
        self.num_spikes = 0
        self.best_energy = 0
        self.iter = 0
        self.thld_iter = 0
        self.thld = 1
        self.t2sota = 0
        self.iter2sota = 0
        self.t2solu = 0
        self.iter2solu = 0
        
        self.v = (self.s_p - self.s_n)
        # self.curr_energy = np.sum(np.sum(np.multiply(Q, 1 - np.outer(self.v, self.v)))) / 4
        self.curr_energy = self.qubo_energy()
    
    def run(self, optimal, debug):
        start = time_ns()
        while self.iter < self.MAX_ITER:
            p = np.random.randint(0, self.N)
            self.update_neuron(p, debug)
            self.iter += 1
            self.thld += self.thld_delta

            if self.best_energy == optimal:
                self.iter2sota = self.iter
                self.t2sota = (time_ns() - start)/1e6

            if debug and self.iter > 200:
                break

            if self.iter > 0 and self.iter % 1e5 == 0:
                print(f'#spikes:{self.num_spikes}, best: {self.best_energy}')
        self.iter2solu = self.iter
        self.t2solu = (time_ns() - start)/1e6

    def update_neuron(self, p, debug):
        s_p_p = self.s_p[p].item()
        s_n_p = self.s_n[p].item()
        vmem_p = self.vmem[p].item()
        
        noisethld = 2.5e4 * math.log(0.99 * random.random() + 1e-6) / (self.thld_max * math.log(1 + self.thld / self.thld_max))
        spike = 1 if (noisethld * (s_p_p - s_n_p) - vmem_p) * (s_p_p - s_n_p) + self.Q[p,p] < 0 else 0
        
        if spike:
            new_ds_p_p = s_n_p > 0
            new_ds_n_p = s_p_p > 0
            self.ds_p[p] = new_ds_p_p
            self.ds_n[p] = new_ds_n_p
            
            self.s_p[p] = 0 if s_p_p else 1
            self.s_n[p] = 0 if s_n_p else 1
            
            # self.curr_energy -= 1 * (new_ds_p_p - new_ds_n_p) * vmem_p
            self.curr_energy += 2*(new_ds_p_p-new_ds_n_p)*vmem_p + self.Q[p,p]

            if debug:
                curr_energy_debug = self.qubo_energy()
                assert curr_energy_debug == self.curr_energy, print(f"{self.curr_energy}, GT: {curr_energy_debug}")

            self.vmem += 1 * (new_ds_p_p - new_ds_n_p) * self.Q[p, :]
            
            if self.curr_energy.item() < self.best_energy:
                self.best_energy = self.curr_energy.item()
            
            self.num_spikes += 1
    
    def qubo_energy(self):
        return self.s_p.T @ self.Q @ self.s_p