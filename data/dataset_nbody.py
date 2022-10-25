from simulate import SimulationDataset
import torch
import numpy as np

# 设置模拟的次数ns（运行很快，可以设大数值）、force种类sim、节点数目n、时间数目nt
sim = 'spring'
ns, nt, n, dim = 10, 1000, 10, 2

# 预定义的六种force下的超参数，包括时间间隔dt、时间数目nt、节点数目n和维度dim
sim_sets = [{'sim': 'r1', 'dt': [5e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'r2', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'spring', 'dt': [1e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'string', 'dt': [1e-2], 'nt': [1000], 'n': [30], 'dim': [2]},
            {'sim': 'charge', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'superposition', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'damped', 'dt': [2e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
            {'sim': 'discontinuous', 'dt': [1e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]}, ]
dt = [ss['dt'][0] for ss in sim_sets if ss['sim'] == sim][0]
print('Running on {}: n={}, dim={}, nt={}, dt={}'.format(sim, n, dim, nt, dt))

s = SimulationDataset(sim, n=n, dim=dim, nt=nt, dt=dt)
s.simulate(ns)

# 以np.array的形式输出，dim=(ns, nt // 2, n, 2 * dim + 2)
data = s.data
acceleration = s.get_acceleration()
potential = s.get_potential()
print(np.array(potential).shape)

# jax的DeviceArray转为np.array后，再转为tensor
data, acceleration, potential = np.array(data), np.array(acceleration), np.array(potential)
torch.save([torch.from_numpy(data), torch.from_numpy(acceleration), torch.from_numpy(potential)], 'hamiltonian.pt')
print('Data shape:', s.data.shape, 'Acceleration shape:', acceleration.shape, 'Potential shape:', potential.shape)

