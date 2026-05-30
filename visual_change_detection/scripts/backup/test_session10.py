import os
from collections import Counter


def count_labels(labels_dir):
    counts = Counter()
    for f in os.listdir(labels_dir):
        if not f.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, f)) as fh:
            for line in fh:
                cls = int(line.strip().split()[0])
                counts[cls] += 1
    return counts


names = ['button', 'text', 'input', 'image', 'icon',
         'checkbox', 'toolbar', 'list_item', 'card', 'menu']
for split in ['train', 'val', 'test']:
    print(f'\n{split}:')
    counts = count_labels(f'./outputs/rico_yolo_dataset_v2/labels/{split}')
    total = sum(counts.values())
    for i, name in enumerate(names):
        n = counts.get(i, 0)
        print(f'  {name:<12} {n:>6}  ({n/total*100:.1f}%)')
    print(f'  TOTAL        {total:>6}')
