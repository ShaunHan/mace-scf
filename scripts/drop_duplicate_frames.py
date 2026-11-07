"""Write the training and validation sets without repeated frames.

    python scripts/drop_duplicate_frames.py TRAIN.xyz VALID.xyz OUT_TRAIN.xyz OUT_VALID.xyz

A repeat is the same calculation stored again: the same atoms at the same positions, total charge, counter
charge and field. One copy of each is kept. A structure with a copy in the validation set keeps one validation
copy and none in training, so that no validation frame is a training frame. The frames are copied as text,
unchanged. The script stops if the energies or Fermi levels of repeats differ by more than TOLERANCE (eV).
"""
import hashlib
import re
import sys
from collections import defaultdict

TOKEN = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
TOLERANCE = 1.e-4


def frames(path):
    with open(path, encoding='utf-8') as handle:
        while True:
            count = handle.readline()
            if not count.strip():
                return
            comment = handle.readline()
            atoms = [handle.readline() for _ in range(int(count))]
            yield [count, comment, *atoms]


def key(block):
    info = {k: v.strip('"') for k, v in TOKEN.findall(block[1]) if len(v) < 200}
    atoms = hashlib.md5('\n'.join(' '.join(line.split()[:4]) for line in block[2:]).encode()).hexdigest()
    labels = tuple(float(info.get(k, 'nan')) for k in ('vasp_free_energy', 'vasp_efermi'))
    return (atoms, info.get('total_charge'), info.get('counter_charge'), info.get('external_field')), labels


def main(train, valid, out_train, out_valid):
    sources = {'train': train, 'valid': valid}
    seen = defaultdict(list)
    for split, path in sources.items():
        for index, block in enumerate(frames(path)):
            name, labels = key(block)
            seen[name].append((split, index, labels))
    keep = {'train': set(), 'valid': set()}
    for name, copies in seen.items():
        first = copies[0][2]
        if any(abs(a-b) > TOLERANCE for _, _, labels in copies for a, b in zip(labels, first)):
            raise ValueError(f'Repeats with different labels: {copies}')
        split, index, _ = next((c for c in copies if c[0] == 'valid'), copies[0])
        keep[split].add(index)
    for split, path, out in (('train', train, out_train), ('valid', valid, out_valid)):
        total = kept = 0
        with open(out, 'w', encoding='utf-8', newline='\n') as sink:
            for index, block in enumerate(frames(path)):
                total += 1
                if index in keep[split]:
                    sink.writelines(block)
                    kept += 1
        print(f'{split}: {total} frames, {kept} kept, {total-kept} dropped')


if __name__ == '__main__':
    main(*sys.argv[1:5])
