"""
Semantic-aware RSU scheduling on CIFAR-10.

Reimplementation and robustness experiments inspired by
"Diversity Maximized Scheduling in RoadSide Units for
Traffic Monitoring Applications."
"""
import csv
import itertools
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
N = 10
M = 3
K = 3
T = 100
NUM_INTERVALS = 25
NUM_CLASSES = 10
SEEDS = list(range(10))
PDR_VALUES = np.arange(0.2, 0.81, 0.1)
ONLINE_AVG_PDR = 0.5
ONLINE_PERCENTAGES = np.array([5, 20, 40, 60, 80, 100], dtype=float)
MIN_CLASSES_PER_RSU = 1
MAX_CLASSES_PER_RSU = 3
MAX_RSUS_PER_CLASS = 2
MODEL_SEED = 999
EPOCHS_PER_INTERVAL = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
UNCERTAINTY_CANDIDATE_MULTIPLIER = 3
MIN_PDR = 0.001
MAX_PDR = 0.999
PDR_SPREAD_FACTOR = 0.95
DELAY_EXP_SCALE = 0.8
DELAY_MINIMUM = 0.2
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
METHODS = ['without_fairness', 'with_fairness', 'uniform', 'random', 'fedcs']
METHOD_NAMES = {'without_fairness': 'optimized without fairness', 'with_fairness': 'optimized using fairness', 'uniform': 'uniform rate', 'random': 'random rate', 'fedcs': 'FedCS'}
MARKERS = {'without_fairness': '>', 'with_fairness': 's', 'uniform': '>', 'random': 'o', 'fedcs': '^'}
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def jain_fairness(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return 1.0
    return float(counts.sum() ** 2 / (len(counts) * np.sum(counts ** 2)))

def normalize(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std

def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))

def choose_rsu_classes(seed):
    """
    Randomly choose 1-3 classes per RSU.

    Conditions:
    - every RSU has at least one class
    - every CIFAR-10 class is represented somewhere
    - no class is assigned to more than MAX_RSUS_PER_CLASS RSUs
    """
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        rsu_classes = []
        for _rsu in range(N):
            number = int(rng.integers(MIN_CLASSES_PER_RSU, MAX_CLASSES_PER_RSU + 1))
            classes = sorted(rng.choice(NUM_CLASSES, size=number, replace=False).tolist())
            rsu_classes.append(classes)
        frequency = np.zeros(NUM_CLASSES, dtype=int)
        for classes in rsu_classes:
            for c in classes:
                frequency[c] += 1
        if np.all(frequency >= 1) and np.all(frequency <= MAX_RSUS_PER_CLASS):
            return rsu_classes
    raise RuntimeError('Could not create a valid random class map.')

def build_random_partition(targets, seed):
    """
    Split the actual 50,000 CIFAR-10 training images among RSUs.

    Image indices do not overlap across RSUs.
    The partition changes when the scenario seed changes.
    """
    rng = np.random.default_rng(seed + 1000)
    rsu_classes = choose_rsu_classes(seed)
    partitions = [[] for _ in range(N)]
    for c in range(NUM_CLASSES):
        owners = [rsu for rsu in range(N) if c in rsu_classes[rsu]]
        indices = np.flatnonzero(targets == c).copy()
        rng.shuffle(indices)
        pieces = np.array_split(indices, len(owners))
        for rsu, piece in zip(owners, pieces):
            partitions[rsu].extend(piece.tolist())
    for rsu in range(N):
        rng.shuffle(partitions[rsu])
    partitions = [np.asarray(x, dtype=np.int64) for x in partitions]
    return (rsu_classes, partitions)

def print_scenario(seed, rsu_classes, partitions, targets):
    print('\n' + '=' * 72)
    print(f'SCENARIO SEED {seed}')
    print('=' * 72)
    for rsu in range(N):
        hist = np.bincount(targets[partitions[rsu]], minlength=NUM_CLASSES)
        info = {CLASS_NAMES[c]: int(hist[c]) for c in rsu_classes[rsu]}
        print(f'RSU {rsu + 1:2d}: {info}')

def generate_delay_history(seed):
    """
    Scenario seed controls heterogeneous 1/lambda_i(t).

    Delay changes across intervals and across RSUs.
    """
    rng = np.random.default_rng(seed + 2000)
    base_delay = rng.exponential(DELAY_EXP_SCALE, size=(NUM_INTERVALS, N)) + DELAY_MINIMUM
    return 1.0 / base_delay

def generate_rsu_pdrs(target_pdr, rng):
    target_pdr = float(target_pdr)
    if not MIN_PDR < target_pdr < MAX_PDR:
        raise ValueError(f'target_pdr must be between {MIN_PDR} and {MAX_PDR}.')
    if not 0.0 < PDR_SPREAD_FACTOR <= 1.0:
        raise ValueError('PDR_SPREAD_FACTOR must be in (0, 1].')
    max_half_range = min(target_pdr - MIN_PDR, MAX_PDR - target_pdr)
    half_range = PDR_SPREAD_FACTOR * max_half_range
    beta = np.linspace(target_pdr - half_range, target_pdr + half_range, N, dtype=float)
    rng.shuffle(beta)
    if beta.min() < MIN_PDR - 1e-12:
        raise RuntimeError('Generated PDR is below MIN_PDR.')
    if beta.max() > MAX_PDR + 1e-12:
        raise RuntimeError('Generated PDR is above MAX_PDR.')
    if not np.isclose(beta.mean(), target_pdr, atol=1e-12):
        raise RuntimeError('Generated PDR mean does not match target PDR.')
    if len(np.unique(np.round(beta, 12))) != N:
        raise RuntimeError('RSU PDR values are not all distinct.')
    return beta

def generate_packet_environment(seed, target_pdr):
    """
    Controlled heterogeneous packet-drop environment.

    For every interval:
    - create N clearly different PDR values
    - keep their mean exactly equal to target_pdr
    - randomly assign them to the N RSUs

    All scheduling methods share the same generated packet events for a
    fair method-to-method comparison inside the same scenario.
    """
    code = int(round(target_pdr * 1000))
    rng = np.random.default_rng(seed * 10000 + code + 3000)
    beta = np.zeros((NUM_INTERVALS, N), dtype=float)
    for interval in range(NUM_INTERVALS):
        beta[interval] = generate_rsu_pdrs(target_pdr, rng)
    success_random = rng.random((NUM_INTERVALS, T, N))
    random_groups = []
    for _ in range(NUM_INTERVALS):
        group = tuple(sorted(rng.choice(N, size=K, replace=False).tolist()))
        random_groups.append(group)
    return (beta, success_random, random_groups)

def print_pdr_snapshot(beta, seed, target_pdr):
    """Print one interval so the controlled heterogeneity is visible."""
    first_interval = beta[0]
    print(f'  PDR snapshot for seed={seed}, target={target_pdr:.1f}, interval=1:')
    for rsu, value in enumerate(first_interval, start=1):
        print(f'    RSU {rsu:2d}: {value:.3f}')
    print(f'    min={first_interval.min():.3f}, max={first_interval.max():.3f}, mean={first_interval.mean():.3f}')

class ImagePools:

    def __init__(self, partitions, targets):
        self.targets = targets
        self.by_rsu_class = []
        for rsu in range(N):
            class_lists = {c: [] for c in range(NUM_CLASSES)}
            for index in partitions[rsu]:
                label = int(targets[index])
                class_lists[label].append(int(index))
            self.by_rsu_class.append(class_lists)
        self.remaining_hist = np.zeros((N, NUM_CLASSES), dtype=int)
        self.refresh_histogram()

    def refresh_histogram(self):
        for rsu in range(N):
            for c in range(NUM_CLASSES):
                self.remaining_hist[rsu, c] = len(self.by_rsu_class[rsu][c])

    def remove_success(self, rsu, image_index):
        label = int(self.targets[image_index])
        pool = self.by_rsu_class[rsu][label]
        try:
            pool.remove(int(image_index))
        except ValueError:
            return
        self.remaining_hist[rsu, label] -= 1
ALL_COALITIONS = list(itertools.combinations(range(N), K))

def effective_delay(lam, beta):
    return 1.0 / (lam * (1.0 - beta))

def choose_optimized(beta, lam, received_counts, remaining_hist, use_fairness):
    f1_values = []
    f2_values = []
    f3_values = []
    for coalition in ALL_COALITIONS:
        idx = np.asarray(coalition, dtype=int)
        delays = effective_delay(lam[idx], beta[idx])
        f1 = 1.0 / delays.sum()
        f2 = float(np.sum(1.0 - beta[idx]))
        predicted = received_counts.astype(float).copy()
        for rsu in coalition:
            hist = remaining_hist[rsu].astype(float)
            total = hist.sum()
            if total == 0:
                continue
            class_probability = hist / total
            expected_successes = T * (1.0 - beta[rsu])
            predicted += expected_successes * class_probability
        f3 = jain_fairness(predicted)
        f1_values.append(f1)
        f2_values.append(f2)
        f3_values.append(f3)
    f1 = normalize(f1_values)
    f2 = normalize(f2_values)
    f3 = normalize(f3_values)
    if use_fairness:
        score = f1 + f2 + f3
    else:
        score = f1 + f2
    return ALL_COALITIONS[int(np.argmax(score))]

def choose_fedcs(lam):
    base_delay = 1.0 / lam
    selected = np.argsort(base_delay)[:K]
    return tuple(sorted(selected.tolist()))

def uniform_active_rsus(slot):
    return tuple(((slot + offset) % N for offset in range(M)))

class Simple8LayerCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.247, 0.2435, 0.2616)
TRAIN_TRANSFORM = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

class IndexedDataset(Dataset):

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base_dataset[int(self.indices[i])]
        return (self.transform(image), label)

def set_model_seed():
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

def build_model(device):
    set_model_seed()
    model = Simple8LayerCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return (model, optimizer)

def train_one_interval(model, optimizer, train_base, received_indices, device):
    if len(received_indices) == 0:
        return
    dataset = IndexedDataset(train_base, received_indices, TRAIN_TRANSFORM)
    generator = torch.Generator().manual_seed(MODEL_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, generator=generator, pin_memory=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS_PER_INTERVAL):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        images = images.to(device)
        prediction = model(images).argmax(dim=1).cpu().numpy()
        y_pred.extend(prediction.tolist())
        y_true.extend(labels.numpy().tolist())
    return macro_f1(y_true, y_pred)

@torch.no_grad()
def score_margins(model, train_base, indices, device):
    """
    Smaller top1-top2 softmax margin means greater uncertainty.
    """
    if len(indices) == 0:
        return np.asarray([], dtype=float)
    images = []
    for index in indices:
        image, _ = train_base[int(index)]
        images.append(EVAL_TRANSFORM(image))
    batch = torch.stack(images).to(device)
    probabilities = torch.softmax(model(batch), dim=1)
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).cpu().numpy()

def allocate_attempts(class_counts, attempts):
    """
    Allocate transmission attempts approximately in proportion to an
    RSU's remaining local class composition.
    """
    counts = np.asarray(class_counts, dtype=float)
    if counts.sum() == 0:
        return np.zeros(NUM_CLASSES, dtype=int)
    raw = attempts * counts / counts.sum()
    allocation = np.floor(raw).astype(int)
    remaining = attempts - allocation.sum()
    order = np.argsort(-(raw - allocation))
    for c in order[:remaining]:
        allocation[c] += 1
    return allocation

def prepare_queue(rsu, attempts, pools, model, train_base, device, seed, interval):
    """
    Keep the local class proportions; within every class choose the
    lowest-margin candidate images.
    """
    allocation = allocate_attempts(pools.remaining_hist[rsu], attempts)
    selected = []
    for c in range(NUM_CLASSES):
        needed = int(allocation[c])
        available = pools.by_rsu_class[rsu][c]
        if needed <= 0 or len(available) == 0:
            continue
        needed = min(needed, len(available))
        candidate_count = min(len(available), max(needed, needed * UNCERTAINTY_CANDIDATE_MULTIPLIER))
        candidates = available[:candidate_count]
        margins = score_margins(model, train_base, candidates, device)
        best = np.argsort(margins)[:needed]
        selected.extend((int(candidates[i]) for i in best))
    rng = np.random.default_rng(seed * 100000 + interval * 100 + rsu)
    rng.shuffle(selected)
    return selected

def run_method(method, seed, partitions, targets, lambda_history, beta_history, success_random, random_groups, train_base, test_loader, device, record_online):
    pools = ImagePools(partitions, targets)
    model, optimizer = build_model(device)
    received_indices = []
    received_counts = np.zeros(NUM_CLASSES, dtype=int)
    cumulative_samples = []
    interval_f1 = []
    for interval in range(NUM_INTERVALS):
        beta = beta_history[interval]
        lam = lambda_history[interval]
        if method == 'without_fairness':
            selected = choose_optimized(beta, lam, received_counts, pools.remaining_hist, False)
        elif method == 'with_fairness':
            selected = choose_optimized(beta, lam, received_counts, pools.remaining_hist, True)
        elif method == 'random':
            selected = random_groups[interval]
        elif method == 'fedcs':
            selected = choose_fedcs(lam)
        else:
            selected = None
        queues = {}
        positions = {}
        if method == 'uniform':
            attempts_per_rsu = T * M // N
            queue_rsus = range(N)
        else:
            attempts_per_rsu = T
            queue_rsus = selected
        for rsu in queue_rsus:
            queues[rsu] = prepare_queue(rsu, attempts_per_rsu, pools, model, train_base, device, seed, interval)
            positions[rsu] = 0
        for slot in range(T):
            if method == 'uniform':
                active_rsus = uniform_active_rsus(slot)
            else:
                active_rsus = selected
            for rsu in active_rsus:
                queue = queues.get(rsu, [])
                position = positions.get(rsu, 0)
                if position >= len(queue):
                    continue
                image_index = queue[position]
                positions[rsu] = position + 1
                if success_random[interval, slot, rsu] >= 1.0 - beta[rsu]:
                    continue
                label = int(targets[image_index])
                received_indices.append(image_index)
                received_counts[label] += 1
                pools.remove_success(rsu, image_index)
        train_one_interval(model, optimizer, train_base, received_indices, device)
        cumulative_samples.append(len(received_indices))
        if record_online:
            interval_f1.append(evaluate_model(model, test_loader, device))
    final_f1 = evaluate_model(model, test_loader, device)
    result = {'final_f1': final_f1, 'received_samples': len(received_indices), 'jain': jain_fairness(received_counts)}
    if record_online:
        cumulative_samples = np.asarray(cumulative_samples, dtype=float)
        interval_f1 = np.asarray(interval_f1, dtype=float)
        final_count = max(cumulative_samples[-1], 1.0)
        stream_percent = 100.0 * cumulative_samples / final_count
        result['online_f1'] = np.interp(ONLINE_PERCENTAGES, stream_percent, interval_f1)
    return result

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n' + '=' * 72)
    print('SEMANTIC-AWARE RSU SCHEDULING')
    print('=' * 72)
    print(f'N={N}, M={M}, K={K}, T={T}, intervals={NUM_INTERVALS}')
    print(f'Scenario seeds: {SEEDS}')
    print(f'Classes per RSU: {MIN_CLASSES_PER_RSU}-{MAX_CLASSES_PER_RSU}')
    print(f'Fixed CNN seed: {MODEL_SEED}')
    print(f'Controlled PDR range: [{MIN_PDR}, {MAX_PDR}], spread factor={PDR_SPREAD_FACTOR}')
    print(f'Device: {device}')
    train_base = CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    test_base = CIFAR10(root=DATA_DIR, train=False, download=True, transform=None)
    targets = np.asarray(train_base.targets)
    test_dataset = IndexedDataset(test_base, np.arange(len(test_base)), EVAL_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=device.type == 'cuda')
    learning = {method: {float(np.round(pdr, 1)): [] for pdr in PDR_VALUES} for method in METHODS}
    online = {method: [] for method in METHODS}
    for seed in SEEDS:
        rsu_classes, partitions = build_random_partition(targets, seed)
        if seed in SEEDS[:2]:
            print_scenario(seed, rsu_classes, partitions, targets)
        lambda_history = generate_delay_history(seed)
        for pdr in PDR_VALUES:
            pdr = float(np.round(pdr, 1))
            beta, success_random, random_groups = generate_packet_environment(seed, pdr)
            print(f'\nSeed {seed:2d} | target avg PDR={pdr:.1f} | actual mean beta={beta.mean():.3f} | global min={beta.min():.3f} | global max={beta.max():.3f}')
            if seed == SEEDS[0]:
                print_pdr_snapshot(beta, seed, pdr)
            record_online = abs(pdr - ONLINE_AVG_PDR) < 1e-09
            for method in METHODS:
                result = run_method(method, seed, partitions, targets, lambda_history, beta, success_random, random_groups, train_base, test_loader, device, record_online)
                learning[method][pdr].append(result['final_f1'])
                if record_online:
                    online[method].append(result['online_f1'])
                print(f"  {METHOD_NAMES[method]:28s} samples={result['received_samples']:4d} Jain={result['jain']:.3f} F1={result['final_f1']:.3f}")
    return (learning, online)

def save_results(learning, online):
    rows = []
    plt.figure(figsize=(7.2, 5.2))
    for method in METHODS:
        means = []
        for pdr in PDR_VALUES:
            pdr = float(np.round(pdr, 1))
            values = np.asarray(learning[method][pdr], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            means.append(mean)
            rows.append({'figure': 'learning_vs_pdr', 'x': pdr, 'method': METHOD_NAMES[method], 'mean_f1': mean, 'std_f1': std, 'num_seeds': len(values)})
        plt.plot(PDR_VALUES, means, marker=MARKERS[method], label=METHOD_NAMES[method])
    plt.xlabel('Average Packet Error Rate')
    plt.ylabel('Learning Accuracy (Macro-F1)')
    plt.title('Learning accuracy vs packet error rate')
    plt.xlim(0.18, 0.82)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'learning_vs_pdr.png', dpi=220)
    plt.close()
    plt.figure(figsize=(7.2, 5.2))
    for method in METHODS:
        values = np.asarray(online[method], dtype=float)
        mean_curve = values.mean(axis=0)
        if len(values) > 1:
            std_curve = values.std(axis=0, ddof=1)
        else:
            std_curve = np.zeros_like(mean_curve)
        plt.plot(ONLINE_PERCENTAGES, mean_curve, marker=MARKERS[method], label=METHOD_NAMES[method])
        for x, mean, std in zip(ONLINE_PERCENTAGES, mean_curve, std_curve):
            rows.append({'figure': 'online_learning', 'x': float(x), 'method': METHOD_NAMES[method], 'mean_f1': float(mean), 'std_f1': float(std), 'num_seeds': len(values)})
    plt.xlabel('Transmitted Dataset [Percentage]')
    plt.ylabel('Learning Accuracy (Macro-F1)')
    plt.title(f'Online learning at average PDR = {ONLINE_AVG_PDR:.1f}')
    plt.xlim(0, 105)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'online_learning.png', dpi=220)
    plt.close()
    with (RESULTS_DIR / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['figure', 'x', 'method', 'mean_f1', 'std_f1', 'num_seeds'])
        writer.writeheader()
        writer.writerows(rows)
    print('\nSaved:')
    print('  results/learning_vs_pdr.png')
    print('  results/online_learning.png')
    print('  results/results.csv')
if __name__ == '__main__':
    learning_results, online_results = run_experiment()
    save_results(learning_results, online_results)
