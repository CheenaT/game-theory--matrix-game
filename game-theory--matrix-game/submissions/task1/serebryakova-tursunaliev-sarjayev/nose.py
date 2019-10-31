import numpy as np
from scipy.optimize import linprog
from nose.tools import assert_equals
import met

def nash_equilibrium1(a):
    c = [-1 for i in range(0,a.shape[1])]
    b = [1 for i in range(0,a.shape[0])]
    q = linprog(c, a, b).x
    p = linprog(b, -a.transpose(),c).x
    opt_sum = 0
    for i in p:
        opt_sum+=i
    cost = 1/opt_sum
    return (p*cost, q*cost,cost)

class TestNash:
    def test_1(self):
        matrix = np.array([
        [8, 4, 7],
        [6, 5, 9],
        [7, 6, 8],
        ])
        
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 6
        p = np.array([0, 0, 1])
        q = np.array([0, 1, 0])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
    def test_2(self):
        matrix = np.array([
        [4, 7, 2],
        [7, 3, 2],
        [2, 1, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 4.02941
        p = np.array([0.42647, 0.23529, 0.33824])
        q = np.array([0.35294, 0.26471, 0.38235])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
    def test_3(self):
        matrix = np.array([
        [3, 7, 0, 9, 1, 4],
        [8, 3, 0, 5, 5, 2],
        [6, 1, 8, 3, 0, 5],
        [8, 7, 2, 0, 6, 9],
        [7, 1, 3, 6, 1, 5],
        [8, 9, 2, 4, 7, 0]
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 3.69795
        p = np.array([0.13856, 0, 0.32918, 0.16642, 0, 0.36584])
        q = np.array([0, 0, 0.24340, 0.30499, 0.28446, 0.16716])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
        
        
        
    def test_4(self):
        matrix = np.array([
        [8, 4, 7],
        [6, 5, 9],
        [7, 6, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=met.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_5(self):
        matrix = np.array([
        [4, 7, 2],
        [7, 3, 2],
        [2, 1, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=met.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_6(self):
        matrix = np.array([
        [3, 7, 0, 9, 1, 4],
        [8, 3, 0, 5, 5, 2],
        [6, 1, 8, 3, 0, 5],
        [8, 7, 2, 0, 6, 9],
        [7, 1, 3, 6, 1, 5],
        [8, 9, 2, 4, 7, 0]
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=met.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
