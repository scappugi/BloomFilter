import matplotlib.pyplot as plt


def plot_scalability(dataset_name, worker_counts, seq_time, mp_std, mp_shm, job_std, job_np):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # 1 riga, 2 colonne, grafico 1

    # Linea orizzontale per il Sequenziale (Baseline)
    plt.axhline(y=seq_time, color='r', linestyle='-', label=f"Sequenziale ({seq_time:.2f}s)")

    # Linee per i paralleli
    plt.plot(worker_counts, mp_std, marker='o', label="MP Standard")
    plt.plot(worker_counts, mp_shm, marker='s', label="MP Shared")
    plt.plot(worker_counts, job_std, marker='^', label="Joblib Std")
    plt.plot(worker_counts, job_np, marker='*', linewidth=2, label="Joblib NumPy (Best)")

    plt.title(f"Tempi di Esecuzione: {dataset_name}")
    plt.xlabel("Numero di Worker")
    plt.ylabel("Tempo (secondi)")
    plt.xticks(worker_counts)  # Mostra tutti i numeri di worker sull'asse
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)  # 1 riga, 2 colonne, grafico 2

    # Calcolo Speedup: T_seq / T_par
    sp_mp_std = [seq_time / t for t in mp_std]
    sp_mp_shm = [seq_time / t for t in mp_shm]
    sp_job_std = [seq_time / t for t in job_std]
    sp_job_np = [seq_time / t for t in job_np]

    # Plot Speedup
    plt.plot(worker_counts, sp_mp_std, marker='o', label="MP Standard")
    plt.plot(worker_counts, sp_mp_shm, marker='s', label="MP Shared")
    plt.plot(worker_counts, sp_job_std, marker='^', label="Joblib Std")
    plt.plot(worker_counts, sp_job_np, marker='*', linewidth=2, label="Joblib NumPy")

    # Linea ideale (Speedup lineare) - Opzionale ma utile
    # Se raddoppio i core, vorrei raddoppiare la velocità (y = x)
    plt.plot(worker_counts, worker_counts, 'k--', alpha=0.3, label="Speedup Ideale (Lineare)")
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)  # Soglia 1x

    plt.title(f"Speedup: {dataset_name}")
    plt.xlabel("Numero di Worker")
    plt.ylabel("Speedup (Nx)")
    plt.xticks(worker_counts)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()