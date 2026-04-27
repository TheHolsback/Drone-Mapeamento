"""
Análise estatística de dados LIDAR
Objetivo: definir filtro mínimo de intensidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import normaltest, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────
FILE_PATH = "./LIDAR.CSV"
OUTPUT_PATH = "./lidar_relatorio.png"
Z_OUTLIER = 2.5   # limiar Z-score para definir "fora da curva"

# ─────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────
df = pd.read_csv(FILE_PATH)
alvos = sorted(df['alvo'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("=" * 65)
print("          RELATÓRIO DE ANÁLISE LIDAR")
print("=" * 65)
print(f"\nDataset: {len(df)} leituras | Alvos: {alvos}\n")

# ─────────────────────────────────────────
# 1. CONTAGEM POR ÂNGULO-ALVO
# ─────────────────────────────────────────
print("━" * 65)
print("1. CONTAGEM DE DADOS POR ALVO")
print("━" * 65)
contagem = df.groupby('alvo').agg(
    n_leituras=('angle', 'count'),
    angulo_min=('angle', 'min'),
    angulo_max=('angle', 'max'),
    cobertura_graus=('angle', lambda x: x.max() - x.min())
).reset_index()
print(contagem.to_string(index=False))

# ─────────────────────────────────────────
# 2. ESTATÍSTICAS DISTÂNCIA POR ALVO
# ─────────────────────────────────────────
print("\n" + "━" * 65)
print("2. DISTRIBUIÇÃO DISTÂNCIA POR ALVO")
print("━" * 65)
dist_stats = {}
for alvo in alvos:
    sub = df[df['alvo'] == alvo]['distance']
    stat, p = normaltest(sub)
    dist_stats[alvo] = {
        'n': len(sub), 'mean': sub.mean(), 'std': sub.std(),
        'median': sub.median(), 'cv_%': sub.std()/sub.mean()*100,
        'skewness': sub.skew(), 'kurtosis': sub.kurtosis(),
        'normal_p': p, 'is_normal': p > 0.05
    }
    print(f"\n  Alvo {int(alvo):3d}°:  n={len(sub)}  "
          f"μ={sub.mean():.0f}mm  σ={sub.std():.0f}mm  "
          f"CV={sub.std()/sub.mean()*100:.1f}%  "
          f"skew={sub.skew():.2f}  kurt={sub.kurtosis():.2f}  "
          f"normal={'✓' if p > 0.05 else '✗'} (p={p:.4f})")

# ─────────────────────────────────────────
# 3. ESTATÍSTICAS INTENSIDADE POR ALVO
# ─────────────────────────────────────────
print("\n" + "━" * 65)
print("3. DISTRIBUIÇÃO INTENSIDADE POR ALVO")
print("━" * 65)
int_stats = {}
for alvo in alvos:
    sub = df[df['alvo'] == alvo]['intensity']
    stat, p = normaltest(sub)
    p5, p10, p25 = sub.quantile(0.05), sub.quantile(0.10), sub.quantile(0.25)
    int_stats[alvo] = {
        'n': len(sub), 'mean': sub.mean(), 'std': sub.std(),
        'p5': p5, 'p10': p10, 'p25': p25,
        'min': sub.min(), 'max': sub.max(),
        'skewness': sub.skew(), 'normal_p': p
    }
    print(f"\n  Alvo {int(alvo):3d}°:  n={len(sub)}  "
          f"μ={sub.mean():.1f}  σ={sub.std():.1f}  "
          f"min={sub.min()}  P5={p5:.0f}  P10={p10:.0f}  P25={p25:.0f}  "
          f"max={sub.max()}  normal={'✓' if p > 0.05 else '✗'} (p={p:.4f})")

# ─────────────────────────────────────────
# 4. CORRELAÇÃO OUTLIER DISTÂNCIA × INTENSIDADE
# ─────────────────────────────────────────
print("\n" + "━" * 65)
print(f"4. CORRELAÇÃO OUTLIER DISTÂNCIA × INTENSIDADE  (|Z| > {Z_OUTLIER})")
print("━" * 65)

df['z_distance'] = np.nan
df['is_outlier'] = False

for alvo in alvos:
    mask = df['alvo'] == alvo
    z = np.abs(stats.zscore(df.loc[mask, 'distance']))
    df.loc[mask, 'z_distance'] = z
    df.loc[mask, 'is_outlier'] = z > Z_OUTLIER

global_corr_r, global_corr_p = pearsonr(df['z_distance'], df['intensity'])
print(f"\n  Correlação global  Z-dist × intensidade:  r={global_corr_r:.3f}  p={global_corr_p:.4f}")

outliers_df = df[df['is_outlier']]
inliers_df  = df[~df['is_outlier']]

print(f"\n  {'Grupo':<12} {'N':>6}  {'Int. μ':>8}  {'Int. σ':>8}  {'Int. min':>9}  {'Int. P5':>8}")
print(f"  {'-'*12} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")
for label, sub in [("Inliers", inliers_df), ("Outliers", outliers_df)]:
    i = sub['intensity']
    print(f"  {label:<12} {len(i):>6}  {i.mean():>8.1f}  {i.std():>8.1f}  {i.min():>9}  {i.quantile(0.05):>8.0f}")

# t-test
t, p_t = stats.ttest_ind(inliers_df['intensity'], outliers_df['intensity'])
print(f"\n  t-test inlier vs outlier:  t={t:.3f}  p={p_t:.4f}  "
      f"→ {'diferença significativa ✓' if p_t < 0.05 else 'sem diferença significativa ✗'}")

print(f"\n  Interpretação:")
if global_corr_r < -0.1 and global_corr_p < 0.05:
    print("  ⚠  Leituras outlier de distância tendem a ter MENOR intensidade — filtragem por")
    print("     intensidade mínima efetivamente reduzirá outliers.")
elif global_corr_r > 0.1 and global_corr_p < 0.05:
    print("  ⚠  Leituras outlier de distância tendem a ter MAIOR intensidade.")
else:
    print("  ✓  Fraca correlação: intensidade e erro de distância são relativamente independentes.")

# ─────────────────────────────────────────
# 5. SUGESTÃO DE HARD FILTER
# ─────────────────────────────────────────
print("\n" + "━" * 65)
print("5. SUGESTÃO DE HARD FILTER DE INTENSIDADE")
print("━" * 65)

thresholds = {}
for alvo in alvos:
    sub = df[df['alvo'] == alvo]['intensity']
    thresholds[alvo] = {
        'conservador (P2)':   int(sub.quantile(0.02)),
        'moderado (P5)':      int(sub.quantile(0.05)),
        'agressivo (P10)':    int(sub.quantile(0.10)),
        'μ - 3σ':             int(max(0, sub.mean() - 3*sub.std())),
        'μ - 2σ':             int(max(0, sub.mean() - 2*sub.std())),
    }

print(f"\n  {'Critério':<20}", end="")
for a in alvos:
    print(f"  Alvo {int(a):3d}°", end="")
print()
print(f"  {'-'*20}", end="")
for _ in alvos:
    print(f"  {'─'*10}", end="")
print()

all_criterios = list(list(thresholds.values())[0].keys())
for crit in all_criterios:
    print(f"  {crit:<20}", end="")
    for a in alvos:
        print(f"  {thresholds[a][crit]:>10}", end="")
    print()

# Recomendação global
all_int = df['intensity']
rec_conserv = int(all_int.quantile(0.02))
rec_moder   = int(all_int.quantile(0.05))
rec_agress  = int(all_int.quantile(0.10))
rec_sigma3  = int(max(0, all_int.mean() - 3*all_int.std()))

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │           RECOMENDAÇÃO GLOBAL (todos os alvos)          │
  ├──────────────────────────────┬──────────────────────────┤
  │ Conservador  (P2  / μ-3σ)   │  {max(rec_conserv,rec_sigma3):>4}  (perde ~2%)       │
  │ Moderado     (P5)            │  {rec_moder:>4}  (perde ~5%)       │
  │ Agressivo    (P10)           │  {rec_agress:>4}  (perde ~10%)      │
  └──────────────────────────────┴──────────────────────────┘

  💡 RECOMENDAÇÃO: usar intensidade_minima = {max(rec_conserv,rec_sigma3)}
     → Remove leituras claramente degradadas sem perder cobertura significativa.
     → Para ambientes mais ruidosos, considere {rec_moder}.
""")

# ─────────────────────────────────────────
# 6. ESTATÍSTICAS ADICIONAIS SUGERIDAS
# ─────────────────────────────────────────
print("━" * 65)
print("6. ESTATÍSTICAS ADICIONAIS")
print("━" * 65)

print("\n  a) Taxa de outliers por alvo:")
for alvo in alvos:
    mask = df['alvo'] == alvo
    n_out = df[mask]['is_outlier'].sum()
    n_tot = mask.sum()
    print(f"     Alvo {int(alvo):3d}°:  {n_out}/{n_tot} outliers  ({100*n_out/n_tot:.1f}%)")

print("\n  b) Leituras com intensidade = 0 (sensor falhou):")
zero_int = df[df['intensity'] == 0]
print(f"     {len(zero_int)} leituras ({100*len(zero_int)/len(df):.2f}%) com intensidade=0")

print("\n  c) Leituras com distância máxima (64999 mm = sem retorno):")
max_dist = df[df['distance'] == 64999]
print(f"     {len(max_dist)} leituras ({100*len(max_dist)/len(df):.2f}%)")
if len(max_dist):
    print(f"     Intensidade média nessas leituras: {max_dist['intensity'].mean():.1f}")

print("\n  d) IQR e valores extremos de intensidade:")
Q1, Q3 = all_int.quantile(0.25), all_int.quantile(0.75)
IQR = Q3 - Q1
fence_low = Q1 - 1.5*IQR
print(f"     IQR={IQR:.1f}  Fence inferior={fence_low:.1f}  "
      f"→ {(all_int < fence_low).sum()} leituras abaixo do fence Tukey")

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor='#0d1117')
fig.suptitle('Análise Estatística LIDAR — Definição de Hard Filter de Intensidade',
             color='white', fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

ax_c   = fig.add_subplot(gs[0, 0])
ax_box = fig.add_subplot(gs[0, 1])
ax_si  = fig.add_subplot(gs[0, 2])
ax_sd  = fig.add_subplot(gs[0, 3])
axes_dist = [fig.add_subplot(gs[1, i]) for i in range(4)]
axes_int  = [fig.add_subplot(gs[2, i]) for i in range(4)]
ax_corr   = fig.add_subplot(gs[3, 0:2])
ax_filt   = fig.add_subplot(gs[3, 2:4])

style = dict(color='white')

# --- contagem por alvo
ax_c.bar([str(int(a)) + '°' for a in alvos],
         [contagem[contagem['alvo']==a]['n_leituras'].values[0] for a in alvos],
         color=colors)
ax_c.set_title('Contagem por Alvo', **style)
ax_c.set_ylabel('N leituras', **style)
ax_c.tick_params(colors='white')
ax_c.set_facecolor('#161b22')
for spine in ax_c.spines.values(): spine.set_edgecolor('#30363d')

# --- boxplot intensidade
bp = ax_box.boxplot(
    [df[df['alvo']==a]['intensity'].values for a in alvos],
    labels=[f'{int(a)}°' for a in alvos],
    patch_artist=True,
    medianprops=dict(color='yellow', linewidth=2)
)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax_box.set_title('Boxplot Intensidade', **style)
ax_box.tick_params(colors='white')
ax_box.set_facecolor('#161b22')
for spine in ax_box.spines.values(): spine.set_edgecolor('#30363d')

# --- scatter intensidade vs distância global
sc = ax_si.scatter(df['distance'], df['intensity'],
                   c=df['alvo'], cmap='Set1', alpha=0.3, s=4)
ax_si.set_title('Distância × Intensidade', **style)
ax_si.set_xlabel('Distância (mm)', **style)
ax_si.set_ylabel('Intensidade', **style)
ax_si.tick_params(colors='white')
ax_si.set_facecolor('#161b22')
for spine in ax_si.spines.values(): spine.set_edgecolor('#30363d')

# --- scatter Z × intensidade
ax_sd.scatter(df['z_distance'], df['intensity'],
              c=['#ff4444' if o else '#44aaff' for o in df['is_outlier']],
              alpha=0.3, s=4)
ax_sd.axvline(Z_OUTLIER, color='yellow', ls='--', lw=1.5, label=f'|Z|={Z_OUTLIER}')
ax_sd.set_title('Z-score Distância × Intensidade', **style)
ax_sd.set_xlabel('|Z-score| distância', **style)
ax_sd.set_ylabel('Intensidade', **style)
ax_sd.legend(fontsize=8, labelcolor='white', facecolor='#161b22')
ax_sd.tick_params(colors='white')
ax_sd.set_facecolor('#161b22')
for spine in ax_sd.spines.values(): spine.set_edgecolor('#30363d')

# --- histogramas distância por alvo
for i, alvo in enumerate(alvos):
    ax = axes_dist[i]
    sub = df[df['alvo']==alvo]['distance']
    sub_clean = sub[sub < 64999]
    ax.hist(sub_clean, bins=40, color=colors[i], alpha=0.7, density=True)
    mu, sigma = sub_clean.mean(), sub_clean.std()
    x = np.linspace(sub_clean.min(), sub_clean.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'white', lw=1.5)
    ax.set_title(f'Distância Alvo {int(alvo)}°\nμ={mu:.0f} σ={sigma:.0f}', **style, fontsize=9)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

# --- histogramas intensidade por alvo
for i, alvo in enumerate(alvos):
    ax = axes_int[i]
    sub = df[df['alvo']==alvo]['intensity']
    ax.hist(sub, bins=40, color=colors[i], alpha=0.7, density=True)
    mu, sigma = sub.mean(), sub.std()
    x = np.linspace(sub.min(), sub.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'white', lw=1.5)
    p5 = sub.quantile(0.05)
    ax.axvline(p5, color='yellow', ls='--', lw=1.5, label=f'P5={p5:.0f}')
    ax.set_title(f'Intensidade Alvo {int(alvo)}°\nμ={mu:.1f} σ={sigma:.1f}', **style, fontsize=9)
    ax.legend(fontsize=7, labelcolor='white', facecolor='#161b22')
    ax.tick_params(colors='white', labelsize=7)
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

# --- correlação outlier: violin
parts = ax_corr.violinplot(
    [inliers_df['intensity'].values, outliers_df['intensity'].values],
    positions=[1, 2], showmedians=True
)
for pc, c in zip(parts['bodies'], ['#44aaff', '#ff4444']):
    pc.set_facecolor(c); pc.set_alpha(0.6)
ax_corr.set_xticks([1, 2])
ax_corr.set_xticklabels(['Inliers', f'Outliers\n|Z|>{Z_OUTLIER}'], color='white')
ax_corr.set_title(f'Intensidade: Inliers vs Outliers (t-test p={p_t:.4f})', **style)
ax_corr.set_ylabel('Intensidade', **style)
ax_corr.tick_params(colors='white')
ax_corr.set_facecolor('#161b22')
for spine in ax_corr.spines.values(): spine.set_edgecolor('#30363d')

# --- curva retenção por threshold
thrs = np.arange(0, 255, 1)
retencao = [(df['intensity'] >= t).mean()*100 for t in thrs]
ax_filt.plot(thrs, retencao, color='#44aaff', lw=2)

limiares = {
    f'Conservador ({max(rec_conserv,rec_sigma3)})': (max(rec_conserv,rec_sigma3), '#90ee90'),
    f'Moderado ({rec_moder})':                       (rec_moder, 'yellow'),
    f'Agressivo ({rec_agress})':                     (rec_agress, '#ff8c00'),
}
for label, (v, c) in limiares.items():
    ret = (df['intensity'] >= v).mean()*100
    ax_filt.axvline(v, color=c, ls='--', lw=1.5)
    ax_filt.annotate(f'{label}\n→ retém {ret:.1f}%',
                     xy=(v, ret), xytext=(v+5, ret-12),
                     color=c, fontsize=7.5,
                     arrowprops=dict(arrowstyle='->', color=c, lw=1))

ax_filt.set_title('Curva de Retenção por Threshold de Intensidade', **style)
ax_filt.set_xlabel('Intensidade mínima', **style)
ax_filt.set_ylabel('% dados retidos', **style)
ax_filt.tick_params(colors='white')
ax_filt.set_facecolor('#161b22')
for spine in ax_filt.spines.values(): spine.set_edgecolor('#30363d')
ax_filt.set_ylim(60, 102)
ax_filt.grid(alpha=0.15, color='white')

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"\n✅ Gráfico salvo em: {OUTPUT_PATH}")
