import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def compute_snr(audio, sr=24000):
    from scipy.signal import butter, filtfilt
    
    nyq = sr / 2
    b, a = butter(4, 5000 / nyq, btype='high')
    noise = filtfilt(b, a, audio)
    
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return 100.0
    
    return 10 * np.log10(signal_power / noise_power)


def compute_zcr(audio):
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    return np.mean(zcr)


def compute_rms_energy(audio):
    rms = librosa.feature.rms(y=audio)[0]
    return np.mean(rms)


def compute_spectral_entropy(audio, sr=24000):
    spec = np.abs(librosa.stft(audio))
    spec_norm = spec / (spec.sum(axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-8), axis=0)
    return np.mean(entropy)


def compute_spectral_centroid(audio, sr=24000):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    return np.mean(centroid)


def compute_spectral_rolloff(audio, sr=24000):
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    return np.mean(rolloff)


def compute_spectral_convergence(audio_real, audio_gen):
    spec_real = np.abs(librosa.stft(audio_real))
    spec_gen = np.abs(librosa.stft(audio_gen))
    
    min_frames = min(spec_real.shape[1], spec_gen.shape[1])
    spec_real = spec_real[:, :min_frames]
    spec_gen = spec_gen[:, :min_frames]
    
    numerator = np.linalg.norm(spec_real - spec_gen, 'fro')
    denominator = np.linalg.norm(spec_real, 'fro')
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_log_spectral_distance(audio_real, audio_gen):
    spec_real = np.abs(librosa.stft(audio_real))
    spec_gen = np.abs(librosa.stft(audio_gen))
    
    min_frames = min(spec_real.shape[1], spec_gen.shape[1])
    spec_real = spec_real[:, :min_frames]
    spec_gen = spec_gen[:, :min_frames]
    
    log_spec_real = np.log(spec_real + 1e-8)
    log_spec_gen = np.log(spec_gen + 1e-8)
    
    return np.sqrt(np.mean((log_spec_real - log_spec_gen) ** 2))


def compute_all_metrics(audio, sr=24000):
    return {
        'snr': compute_snr(audio, sr),
        'zcr': compute_zcr(audio),
        'rms_energy': compute_rms_energy(audio),
        'spectral_entropy': compute_spectral_entropy(audio, sr),
        'spectral_centroid': compute_spectral_centroid(audio, sr),
        'spectral_rolloff': compute_spectral_rolloff(audio, sr),
        'duration': len(audio) / sr
    }


def evaluate_audio_quality(real_dir, gen_dir, sr=24000, num_samples=100):
    real_dir = Path(real_dir)
    gen_dir = Path(gen_dir)
    
    real_files = sorted(list(real_dir.glob('*.wav')))[:num_samples]
    gen_files = sorted(list(gen_dir.glob('*.wav')))[:num_samples]
    
    print(f"Found {len(real_files)} real files and {len(gen_files)} generated files")
    
    real_metrics = {
        'snr': [], 'zcr': [], 'rms_energy': [],
        'spectral_entropy': [], 'spectral_centroid': [],
        'spectral_rolloff': [], 'duration': []
    }
    
    print("\nComputing metrics for real audio...")
    for audio_file in tqdm(real_files):
        audio, _ = librosa.load(audio_file, sr=sr, mono=True)
        metrics = compute_all_metrics(audio, sr)
        
        for key, value in metrics.items():
            real_metrics[key].append(value)
    
    gen_metrics = {
        'snr': [], 'zcr': [], 'rms_energy': [],
        'spectral_entropy': [], 'spectral_centroid': [],
        'spectral_rolloff': [], 'duration': []
    }
    
    print("\nComputing metrics for generated audio...")
    for audio_file in tqdm(gen_files):
        audio, _ = librosa.load(audio_file, sr=sr, mono=True)
        metrics = compute_all_metrics(audio, sr)
        
        for key, value in metrics.items():
            gen_metrics[key].append(value)
    
    print("\nComputing pairwise similarity metrics...")
    spectral_convergence = []
    log_spectral_distance = []
    
    num_pairs = min(len(real_files), len(gen_files))
    for i in tqdm(range(num_pairs)):
        real_audio, _ = librosa.load(real_files[i], sr=sr, mono=True)
        gen_audio, _ = librosa.load(gen_files[i], sr=sr, mono=True)
        
        sc = compute_spectral_convergence(real_audio, gen_audio)
        lsd = compute_log_spectral_distance(real_audio, gen_audio)
        
        spectral_convergence.append(sc)
        log_spectral_distance.append(lsd)
    
    results = {}
    
    for key in real_metrics.keys():
        real_vals = np.array(real_metrics[key])
        gen_vals = np.array(gen_metrics[key])
        
        results[key] = {
            'real_mean': float(np.mean(real_vals)),
            'real_std': float(np.std(real_vals)),
            'gen_mean': float(np.mean(gen_vals)),
            'gen_std': float(np.std(gen_vals)),
            'difference_pct': float((np.mean(gen_vals) - np.mean(real_vals)) / np.mean(real_vals) * 100)
        }
        
        ks_stat, p_value = stats.ks_2samp(real_vals, gen_vals)
        results[key]['ks_statistic'] = float(ks_stat)
        results[key]['p_value'] = float(p_value)
    
    results['spectral_convergence'] = {
        'mean': float(np.mean(spectral_convergence)),
        'std': float(np.std(spectral_convergence))
    }
    
    results['log_spectral_distance'] = {
        'mean': float(np.mean(log_spectral_distance)),
        'std': float(np.std(log_spectral_distance))
    }
    
    print("\n" + "="*60)
    print("ACOUSTIC QUALITY EVALUATION")
    print("="*60)
    
    for key in ['snr', 'zcr', 'rms_energy', 'spectral_entropy', 
                'spectral_centroid', 'spectral_rolloff']:
        r = results[key]
        print(f"\n{key.upper()}:")
        print(f"  Real:      {r['real_mean']:.4f} ± {r['real_std']:.4f}")
        print(f"  Generated: {r['gen_mean']:.4f} ± {r['gen_std']:.4f}")
        print(f"  Difference: {r['difference_pct']:+.2f}%")
        print(f"  KS test: D={r['ks_statistic']:.3f}, p={r['p_value']:.4f}")
    
    print(f"\nSPECTRAL CONVERGENCE:")
    print(f"  {results['spectral_convergence']['mean']:.4f} ± {results['spectral_convergence']['std']:.4f}")
    
    print(f"\nLOG SPECTRAL DISTANCE:")
    print(f"  {results['log_spectral_distance']['mean']:.4f} ± {results['log_spectral_distance']['std']:.4f}")
    
    print("="*60)
    
    return results, real_metrics, gen_metrics


def plot_distributions(real_metrics, gen_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['snr', 'zcr', 'rms_energy', 'spectral_entropy',
                       'spectral_centroid', 'spectral_rolloff']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        ax.hist(real_metrics[metric], bins=30, alpha=0.6, label='Real', color='blue')
        ax.hist(gen_metrics[metric], bins=30, alpha=0.6, label='Generated', color='red')
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'metric_distributions.png'}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        data = [real_metrics[metric], gen_metrics[metric]]
        bp = ax.boxplot(data, labels=['Real', 'Generated'], patch_artist=True)
        
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'metric_boxplots.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--real_audio", type=str, required=True)
    parser.add_argument("--generated_audio", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    results, real_metrics, gen_metrics = evaluate_audio_quality(
        args.real_audio,
        args.generated_audio,
        sr=args.sample_rate,
        num_samples=args.num_samples
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'evaluation_results.json'}")
    
    plot_distributions(real_metrics, gen_metrics, output_dir)
