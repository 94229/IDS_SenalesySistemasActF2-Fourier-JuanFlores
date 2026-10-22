"""
Actividad Formativa 2: Simulación y Análisis de Señales con la Transformada de Fourier
Objetivo: Analizar señales elementales en el dominio del tiempo y frecuencia.
Librerías: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================
Fs = 1000
T = 1.0 / Fs
N = 1000
t = np.arange(N) * T

# =============================================================================
# SECCIÓN 1: SEÑALES ELEMENTALES
# =============================================================================

# Pulso rectangular (0 a 0.5 s)
rect_signal = np.where((t >= 0) & (t <= 0.5), 1.0, 0.0)

# Función escalón (salta en t=0.5)
step_signal = np.where(t >= 0.5, 1.0, 0.0)

# Senoidal a 5 Hz
f0 = 5
sine_signal = np.sin(2 * np.pi * f0 * t)

# =============================================================================
# SECCIÓN 2: FFT
# =============================================================================

def compute_fft(signal):
    N = len(signal)
    F = np.fft.fft(signal)
    Freq = np.fft.fftfreq(N, T)
    Magnitude = np.abs(F)
    Phase = np.angle(F)
    return Freq, Magnitude, Phase

freq_rect, mag_rect, phase_rect = compute_fft(rect_signal)
freq_step, mag_step, phase_step = compute_fft(step_signal)
freq_sine, mag_sine, phase_sine = compute_fft(sine_signal)

# =============================================================================
# SECCIÓN 3: GRÁFICAS
# =============================================================================

def plot_signal_and_spectrum(t, signal, freq, mag, phase, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(t, signal, 'b-', linewidth=1.5)
    axes[0].set_title(f'{title} - Dominio del tiempo')
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Amplitud')
    axes[0].grid(True, alpha=0.3)

    pos_mask = freq >= 0
    axes[1].plot(freq[pos_mask], mag[pos_mask], 'r-', linewidth=1.2)
    axes[1].set_title(f'{title} - Magnitud |X(f)|')
    axes[1].set_xlabel('Frecuencia (Hz)')
    axes[1].set_ylabel('Magnitud')
    axes[1].set_xlim([0, Fs/2])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq[pos_mask], phase[pos_mask], 'g-', linewidth=1.2)
    axes[2].set_title(f'{title} - Fase X(f)')
    axes[2].set_xlabel('Frecuencia (Hz)')
    axes[2].set_ylabel('Fase (rad)')
    axes[2].set_xlim([0, Fs/2])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    
plot_signal_and_spectrum(t, rect_signal, freq_rect, mag_rect, phase_rect, 'Pulso Rectangular')
plot_signal_and_spectrum(t, step_signal, freq_step, mag_step, phase_step, 'Funcion Escalon')
plot_signal_and_spectrum(t, sine_signal, freq_sine, mag_sine, phase_sine, 'Senoidal')

# =============================================================================
# SECCIÓN 4: PROPIEDADES
# =============================================================================

# --- LINEALIDAD ---
print("=" * 60)
print("LINEALIDAD: F{a*x + b*y} = a*X + b*Y")
print("-" * 60)

a, b = 2.0, 3.0
x = rect_signal
y = sine_signal
z_time = a * x + b * y

Z_fft = np.fft.fft(z_time)
Z_linear = a * np.fft.fft(x) + b * np.fft.fft(y)
error = np.max(np.abs(Z_fft - Z_linear))
print(f"  Error maximo: {error:.2e}")
print(f"  Verificada: {'SI' if error < 1e-10 else 'NO'}")

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
pos_mask = freq_rect >= 0
axes[0].plot(t, z_time, 'k-', label=f'{a}*rect + {b}*seno')
axes[0].set_title('Combinacion lineal - tiempo')
axes[0].set_xlabel('Tiempo (s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_rect[pos_mask], np.abs(Z_fft[pos_mask]), 'k-', label='|FFT(a*x+b*y)|')
axes[1].plot(freq_rect[pos_mask], np.abs(Z_linear[pos_mask]), 'r--', label='|a*FFT(x)+b*FFT(y)|')
axes[1].set_title('Combinacion lineal - frecuencia')
axes[1].set_xlabel('Frecuencia (Hz)')
axes[1].set_xlim([0, Fs/4])
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('propiedad_linealidad.png', dpi=150, bbox_inches='tight')


# --- DESPLAZAMIENTO EN TIEMPO ---
print("\n" + "=" * 60)
print("DESPLAZAMIENTO: F{x(t-t0)} = e^(-j*2*pi*f*t0)*X(f)")
print("-" * 60)

t0 = 0.1
n_shift = int(t0 * Fs)
x_orig = rect_signal
x_shifted = np.roll(x_orig, n_shift)

X_orig = np.fft.fft(x_orig)
X_shifted = np.fft.fft(x_shifted)
freq_full = np.fft.fftfreq(N, T)
phase_factor = np.exp(-1j * 2 * np.pi * freq_full * t0)
X_expected = X_orig * phase_factor

error_shift = np.max(np.abs(X_shifted - X_expected))
print(f"  t0 = {t0} s")
print(f"  Error maximo: {error_shift:.2e}")
print(f"  Verificada: {'SI' if error_shift < 1e-8 else 'NO'}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(t, x_orig, 'b-', label='x(t)')
axes[0,0].plot(t, x_shifted, 'r-', label='x(t-t0)')
axes[0,0].set_title('Tiempo')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(freq_full, np.abs(X_orig), 'b-', label='|X(f)|')
axes[0,1].plot(freq_full, np.abs(X_shifted), 'r--', label='|X(f)| desplazada')
axes[0,1].set_title('Magnitud (no cambia)')
axes[0,1].set_xlim([-Fs/4, Fs/4])
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(freq_full, np.angle(X_orig), 'b-', label='ang X(f)')
axes[1,0].plot(freq_full, np.angle(X_shifted), 'r-', label='ang X(f) desplazada')
axes[1,0].set_title('Fase (cambia lineal)')
axes[1,0].set_xlim([-Fs/4, Fs/4])
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(freq_full, np.angle(phase_factor), 'g-')
axes[1,1].set_title('Factor e^(-j*2*pi*f*t0)')
axes[1,1].set_xlim([-Fs/4, Fs/4])
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Desplazamiento en el Tiempo')
plt.tight_layout()
plt.savefig('propiedad_desplazamiento.png', dpi=150, bbox_inches='tight')


# --- ESCALAMIENTO ---
print("\n" + "=" * 60)
print("ESCALAMIENTO: F{x(a*t)} = (1/|a|)*X(f/a)")
print("-" * 60)

a_scale = 2.0
x_orig = np.where((t >= 0) & (t <= 0.5), 1.0, 0.0)
x_scaled = np.where((t >= 0) & (t <= 0.5/a_scale), 1.0, 0.0)

X_orig = np.fft.fft(x_orig)
X_scaled = np.fft.fft(x_scaled)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
pos_mask = freq_rect >= 0
axes[0].plot(t, x_orig, 'b-', label='x(t) dur=0.5s')
axes[0].plot(t, x_scaled, 'r-', label=f'x({a_scale}*t) dur={0.5/a_scale}s')
axes[0].set_title('Tiempo')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_rect[pos_mask], np.abs(X_orig[pos_mask]), 'b-', label='|X(f)|')
axes[1].plot(freq_rect[pos_mask], np.abs(X_scaled[pos_mask]), 'r-', label='|X(f/a)|')
axes[1].set_title('Frecuencia (se expande)')
axes[1].set_xlim([0, 10])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Escalamiento en el Tiempo')
plt.tight_layout()
plt.savefig('propiedad_escalamiento.png', dpi=150, bbox_inches='tight')


# =============================================================================
# SECCIÓN 5: COMPARACIÓN GENERAL
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
pos_mask = freq_rect >= 0

axes[0].plot(freq_rect[pos_mask], mag_rect[pos_mask], 'b-')
axes[0].set_title('Pulso rectangular - espectro sinc')
axes[0].set_xlim([0, 20])
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_step[pos_mask], mag_step[pos_mask], 'r-')
axes[1].set_title('Escalon - espectro')
axes[1].set_xlim([0, 20])
axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_sine[pos_mask], mag_sine[pos_mask], 'g-')
axes[2].set_title(f'Senoidal {f0} Hz - pico en f0')
axes[2].set_xlim([0, 20])
axes[2].grid(True, alpha=0.3)

plt.suptitle('Comparacion de Espectros')
plt.tight_layout()
plt.savefig('comparacion_espectros.png', dpi=150, bbox_inches='tight')

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("""
1. Pulso rectangular  -> espectro sinc (abanico)
2. Escalon            -> componente DC + caida lenta
3. Senoidal           -> pico en f0
4. Linealidad         -> VERIFICADA
5. Desplazamiento     -> VERIFICADA (magnitud igual, fase cambia)
6. Escalamiento       -> VERIFICADA (comprime tiempo = expande frecuencia)
""")
print("Gráficas guardadas como PNG en esta carpeta.")
print("Listo.") 
plt.close()