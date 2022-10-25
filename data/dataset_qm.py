from scipy import io
import pandas as pd
from time import time
import torch
from torch.nn.utils.rnn import pad_sequence
from atom3d.datasets import LMDBDataset
import atom3d.util.formats as fo
from ase.units import Ha, eV

element = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def qm7():
    data = io.loadmat('./qm7.mat')
    x = torch.tensor(data['Z'])
    pos = torch.tensor(data['R'])
    y = torch.tensor(data['T'])[0]

    # 加入占位原子
    x = torch.cat((torch.ones(x.shape[0]).unsqueeze(-1) * 20, x), dim=1)
    pos = torch.cat((torch.mean(pos, dim=-2, keepdim=True), pos), dim=1)
    assert x.shape[0] == y.shape[0], 'Some samples are missed!'
    torch.save([x, pos, y], '../data/qm7.pt')


def qm8():
    y = pd.read_csv('../../qm8.sdf.csv')
    # 取第一个指标
    y = torch.tensor(y.to_numpy())[:, 1]
    x, pos = [], []
    start = time()
    with open('../../qm8.sdf') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            if lines[idx][0] == 'g':
                if idx > 0:
                    x.append(torch.tensor(tmp_x))
                    pos.append(torch.tensor(tmp_pos))
                tmp_x, tmp_pos = [], []
                idx += 4
            atom = lines[idx].split()
            if len(atom) > 15:
                tmp_x.append([element[i] for i in atom[3]])
                tmp_pos.append([float(i) for i in atom[:3]])
            idx += 1

    # 加上最后一个
    x.append(torch.tensor(tmp_x))
    pos.append(torch.tensor(tmp_pos))

    x = pad_sequence(x, batch_first=True, padding_value=0)[..., 0]
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    torch.save([x, pos, y], '../data/qm8.pt')
    print('Finished with {} data with {}s.'.format(len(x), time() - start))
    assert len(x) == y.shape[0], 'Some samples are missed!'


def qm9():
    dataset = LMDBDataset('../../raw/QM9/data/')
    x, pos = [], []
    for i in range(len(dataset)):
        if i % 4000 == 0:
            print(f'Currently processed at {i}')
        atoms = dataset[i]['atoms']
        x_tmp = []
        for m in atoms['element']:
            x_tmp.append(element[m])
        x.append(torch.tensor(x_tmp))
        pos.append(torch.tensor(fo.get_coordinates_from_df(atoms)))

    y = torch.tensor([item['labels'] for item in dataset])[:, 3:15]
    y[:, 2:5] *= Ha / eV
    y[:, 6] *= Ha / eV * 1000
    y[:, 7:11] *= Ha / eV
    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    torch.save([x, pos, y], '../data/qm9.pt')
    print(f'Loading {len(x) / 1000}K data samples with {len(element)} atom types.')


if __name__ == '__main__':
    qm8()
