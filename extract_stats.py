import re

def parse_log(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        return

    # Normalize multiline entries (training logs are often multi-line)
    # They usually start with a timestamp
    entries = re.split(r'\n(?=\d{4}-\d{2}-\d{2})', content)

    results = []
    last_train_stats = None
    
    # Regex for training stats
    # Iters: 79990/80000 ... T_mean: 0.9951 ... T_min: 0.9000 || T_max: 2.1000 || mc(mean/min/max): 0.9795...
    # || r~(mean/min/max): 1.7734e-01/ ... || a(mean/min/max): 83093.0547/ ...
    # || |dr/dT|: 6.07e+00 || conv<1e-03: 0.0% || |T-T0|: 0.547
    
    train_re = re.compile(
        r'Iters: (\d+)/\d+ .*?'
        r'T_mean: ([\d.]+) .*?'
        r'T_min: ([\d.]+) .*?'
        r'T_max: ([\d.]+) .*?'
        r'mc\(mean/min/max\): ([\d.e+-]+).*?'
        r'r~\(mean/min/max\): ([\d.e+-]+).*?'
        r'a\(mean/min/max\): ([\d.e+-]+).*?'
        r'\|dr/dT\|: ([\d.e+-]+) .*?'
        r'conv<1e-03: ([\d.%]+) .*?'
        r'\|T-T0\|: ([\d.]+)',
        re.DOTALL
    )
    
    # Regex for validation stats
    # Sample: 1449, Validation pixAcc: 0.908, mIoU: 0.631
    val_re = re.compile(r'Sample: 1449, Validation pixAcc: ([\d.]+), mIoU: ([\d.]+)')

    for entry in entries:
        train_match = train_re.search(entry)
        if train_match:
            last_train_stats = list(train_match.groups())
            # Replace percentage in conv if present
            last_train_stats[8] = last_train_stats[8].replace('%', '')
        
        val_match = val_re.search(entry)
        if val_match and last_train_stats:
            iter_num = last_train_stats[0]
            val_pixAcc = val_match.group(1)
            val_mIoU = val_match.group(2)
            results.append([iter_num] + last_train_stats[1:] + [val_pixAcc, val_mIoU])

    header = ["Iter", "T_mean", "T_min", "T_max", "mc", "r~", "a", "|dr/dT|", "conv", "|T-T0|", "pixAcc", "mIoU"]
    print("\t".join(header))
    for r in results:
        print("\t".join(r))

if __name__ == "__main__":
    parse_log("/workspace/covar+cirkd/CIRKD-main/train.log")
