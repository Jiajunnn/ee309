import argparse

RANGES = {
    "energy": (0.105, 1107),       # pJ/bit, lower is better
    "area": (0.000135, 1.4),       # um^2
    "latency": (0.5, 662500),      # ns
    "bandwidth": (0.0865, 1024),   # GB/s, higher is better
    "endurance": (1e4, 1e16),      # cycles
    "ser": (0, 5100),              # FIT/Mb, lower is better
    "retention": (0, 20),          # years
    "temp": (0, 150),              # °C (max)
}

DEVICES = [
    {
        "name": "SRAM",
        "placement": "on",
        "metrics": {
            "energy": 1107,
            "area": 0.117,
            "latency": 0.5,
            "bandwidth": 50,
            "endurance": 1e16,
            "ser": 5100,
            "retention": 0,
            "temp": 0,
        },
    },
    {
        "name": "DRAM",
        "placement": "off",
        "metrics": {
            "energy": 22.25,
            "area": 0.006,
            "latency": 75,
            "bandwidth": 1024,
            "endurance": 1e16,
            "ser": 500,
            "retention": 0,
            "temp": 85,
        },
    },
    {
        "name": "RRAM",
        "placement": "both",
        "metrics": {
            "energy": 0.55,
            "area": 0.1,
            "latency": 5,
            "bandwidth": 1,
            "endurance": 1e9,
            "ser": 0,
            "retention": 10,
            "temp": 150,
        },
    },
    {
        "name": "Gain Cell",
        "placement": "on",
        "metrics": {
            "energy": 3.45,
            "area": 0.08,
            "latency": 4.5,
            "bandwidth": 83,
            "endurance": 1e14,
            "ser": 1100,
            "retention": 0,
            "temp": 80,
        },
    },
    {
        "name": "PCM",
        "placement": "both",
        "metrics": {
            "energy": 1.1,
            "area": 0.0009415,
            "latency": 105,
            "bandwidth": 0.0865,
            "endurance": 1e9,
            "ser": 0,
            "retention": 10,
            "temp": 120,
        },
    },
    {
        "name": "FeRAM",
        "placement": "both",
        "metrics": {
            "energy": 0.105,
            "area": 1.4,
            "latency": 105,
            "bandwidth": 0.1,
            "endurance": 1.5e11,
            "ser": 0,
            "retention": 20,
            "temp": 100,
        },
    },
    {
        "name": "NAND Flash",
        "placement": "off",
        "metrics": {
            "energy": 935,
            "area": 0.000135,
            "latency": 662500,
            "bandwidth": 0.6,
            "endurance": 1e4,
            "ser": 0,
            "retention": 10,
            "temp": 87.5,
        },
    },
    {
        "name": "STT-MRAM",
        "placement": "both",
        "metrics": {
            "energy": 5.1,
            "area": 0.17,
            "latency": 15,
            "bandwidth": 0.1,
            "endurance": 1e13,
            "ser": 0,
            "retention": 20,
            "temp": 125,
        },
    },
]

WORKLOAD_WEIGHTS = {
    "on": {
        "mobile": {"latency": 0.45, "endurance": 0.35, "retention": 0.15, "energy": 0.05},
        "ai": {"latency": 0.4, "endurance": 0.3, "bandwidth": 0.3},
        "ext": {"temp": 0.5, "retention": 0.3, "ser": 0.2},
    },
    "off": {
        "mobile": {"area": 0.7, "bandwidth": 0.2, "retention": 0.1},
        "ai": {"bandwidth": 0.6, "area": 0.4},
        "ext": {"temp": 0.3, "energy": 0.3, "area": 0.2, "retention": 0.2},
    },
}

WORKLOAD_LABELS = {
    "mobile": "Mobile / Embedded",
    "ai": "AI / HPC",
    "ext": "Extreme Environment",
}

def normalize(value, vmin, vmax, higher_is_better=True):
    if vmax == vmin:
        return 5.0
    if higher_is_better:
        score = (value - vmin) / (vmax - vmin)
    else:
        score = (vmax - value) / (vmax - vmin)
    return 1 + 9 * max(0, min(1, score))


def metric_score(metric, value):
    vmin, vmax = RANGES[metric]
    higher_is_better = metric in {"bandwidth", "endurance", "retention", "temp"}
    return normalize(value, vmin, vmax, higher_is_better=higher_is_better)


def score_device(device):
    per_metric = {m: metric_score(m, v) for m, v in device["metrics"].items()}
    return per_metric


def evaluate_devices(placement):
    if placement == "off":
        candidates = [d for d in DEVICES if d["placement"] in {"off", "both"}]
    else:
        candidates = [d for d in DEVICES if d["placement"] in {"on", "both"}]

    results = []
    for device in candidates:
        per_metric = score_device(device)
        results.append({"name": device["name"], "metrics": per_metric})
    rankings = {}
    for workload, weights in WORKLOAD_WEIGHTS[placement].items():
        scored = []
        for entry in results:
            score = sum(entry["metrics"][m] * weight for m, weight in weights.items())
            workloads = entry.get("workloads") or {}
            workloads[workload] = score
            entry["workloads"] = workloads
            scored.append(entry)
        rankings[workload] = sorted(scored, key=lambda item: item["workloads"][workload], reverse=True)
    return rankings


def sensitivity_analysis(placement, perturbation=0.1):
    """Run sensitivity analysis by perturbing weights and checking ranking stability."""
    import random
    
    baseline_rankings = evaluate_devices(placement)
    
    print(f"\n\n=== SENSITIVITY ANALYSIS (±{int(perturbation*100)}% weight variation) ===")
    
    num_trials = 100
    ranking_changes = {workload: {device: [] for device in range(1, 9)} for workload in WORKLOAD_LABELS.keys()}
    
    for trial in range(num_trials):
        # Perturb weights
        perturbed_weights = {}
        for workload, weights in WORKLOAD_WEIGHTS[placement].items():
            new_weights = {}
            total = 0
            for metric, weight in weights.items():
                # Random perturbation between -perturbation and +perturbation
                factor = 1 + random.uniform(-perturbation, perturbation)
                new_weights[metric] = weight * factor
                total += new_weights[metric]
            # Renormalize to sum to 1
            new_weights = {m: w/total for m, w in new_weights.items()}
            perturbed_weights[workload] = new_weights
        
        # Temporarily replace weights
        original_weights = WORKLOAD_WEIGHTS[placement].copy()
        WORKLOAD_WEIGHTS[placement] = perturbed_weights
        
        # Evaluate with perturbed weights
        perturbed_rankings = evaluate_devices(placement)
        
        # Restore original weights
        WORKLOAD_WEIGHTS[placement] = original_weights
        
        # Track ranking positions
        for workload in WORKLOAD_LABELS.keys():
            for rank, entry in enumerate(perturbed_rankings[workload], start=1):
                ranking_changes[workload][rank].append(entry['name'])
    
    # Analyze results
    for workload_key, label in WORKLOAD_LABELS.items():
        print(f"\n{label}:")
        baseline = baseline_rankings[workload_key]
        
        for rank in range(1, min(4, len(baseline) + 1)):
            baseline_device = baseline[rank - 1]['name']
            alternatives = ranking_changes[workload_key][rank]
            device_counts = {}
            for device in alternatives:
                device_counts[device] = device_counts.get(device, 0) + 1
            
            # Sort by frequency
            sorted_devices = sorted(device_counts.items(), key=lambda x: x[1], reverse=True)
            
            print(f"  Rank #{rank} (baseline: {baseline_device}):")
            for device, count in sorted_devices[:3]:
                percentage = (count / num_trials) * 100
                stability = "STABLE" if device == baseline_device and percentage > 80 else ""
                print(f"    {device}: {percentage:.1f}% {stability}")


def main():
    parser = argparse.ArgumentParser(description="Rank memory devices for on/off-chip use cases.")
    parser.add_argument(
        "--placement",
        choices=("on", "off"),
        default="off",
        help="Select the hierarchy (on or off-chip). Default: off",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis by varying weights",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.15,
        help="Weight perturbation factor for sensitivity analysis (default: 0.15 = ±15%%)",
    )
    args = parser.parse_args()

    rankings = evaluate_devices(args.placement)

    if not rankings:
        print(f"No devices available for {args.placement}-chip placement.")
        return

    print(f"\n=== Top Memory Devices for {args.placement.upper()}-CHIP ===")
    for workload_key, label in WORKLOAD_LABELS.items():
        ranking = rankings[workload_key][:3]
        if not ranking:
            continue
        print(f"\n{label}:")
        for idx, entry in enumerate(ranking, start=1):
            workloads = entry["workloads"]
            print(f" {idx}. {entry['name']}: {label} score {workloads[workload_key]:.2f}")
            print(
                "    Mobile {:.2f} | AI {:.2f} | Extreme {:.2f}".format(
                    workloads["mobile"], workloads["ai"], workloads["ext"]
                )
            )
    
    # Print all device scores for each workload
    print(f"\n\n=== All Device Scores for {args.placement.upper()}-CHIP ===")
    for workload_key, label in WORKLOAD_LABELS.items():
        all_devices = rankings[workload_key]
        print(f"\n{label} - All Devices:")
        for idx, entry in enumerate(all_devices, start=1):
            workloads = entry["workloads"]
            print(f" {idx}. {entry['name']}: {workloads[workload_key]:.2f}")
    
    # Run sensitivity analysis if requested
    if args.sensitivity:
        sensitivity_analysis(args.placement, args.perturbation)


if __name__ == "__main__":
    main()