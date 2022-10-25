from ase.db import connect
import torch


x, pos, y = [], [], []
# 80%的MD steps
with connect('../../iso17/reference.db') as conn:
    # 可以设置select(limit=1)做测试
    for row in conn.select():
        x.append(torch.tensor(row['numbers']))
        pos.append(torch.tensor(row['positions']))
        y.append(torch.tensor(row.data['atomic_forces']))
        if len(x) % 1000 == 0:
            print(f'Currently processing at {len(x) / 1000}K samples.')

# 剩余的20% steps
with connect('../../iso17/test_within.db') as conn:
    for row in conn.select():
        x.append(torch.tensor(row['numbers']))
        pos.append(torch.tensor(row['positions']))
        y.append(torch.tensor(row.data['atomic_forces']))
        if len(x) % 1000 == 0:
            print(f'Currently processing at {len(x) / 1000}K samples.')

x = torch.stack(x, dim=0)
pos = torch.stack(pos, dim=0)
y = torch.stack(y, dim=0)

torch.save([x, pos, y], 'md.pt')










