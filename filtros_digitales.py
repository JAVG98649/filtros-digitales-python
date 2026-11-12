"""
Actividad: Implementación y evaluación de filtros digitales
Descripción: Diseño, aplicación y evaluación de filtros digitales Pasa Bajos (LPF),
             Pasa Altos (HPF) y Pasa Bandas (BPF) utilizando arquitecturas IIR
             (Butterworth) y FIR (Ventana de Hamming) sobre señales compuestas con ruido.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

# ==============================================================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==============================================================================
fs = 1000.0               # Frecuencia de muestreo (Hz)
nyq = 0.5 * fs            # Frecuencia de Nyquist (500 Hz)
duracion = 1.0            # Duración de la señal en segundos
t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)

# Frecuencias fundamentales de prueba
f_baja = 15.0             # Componente de baja frecuencia (15 Hz)
f_media = 120.0           # Componente de media frecuencia (120 Hz)
f_alta = 300.0            # Componente de alta frecuencia (300 Hz)

# Señal limpia (suma de las tres sinusoides)
senal_limpia = (1.0 * np.sin(2 * np.pi * f_baja * t) +
                0.8 * np.sin(2 * np.pi * f_media * t) +
                0.6 * np.sin(2 * np.pi * f_alta * t))

# Adición de ruido blanco gaussiano
np.random.seed(42)
ruido = np.random.normal(0, 0.7, len(t))
senal_ruidosa = senal_limpia + ruido


# ==============================================================================
# 2. DISEÑO DE LOS FILTROS DIGITALES (IIR BUTTERWORTH Y FIR VENTANA HAMMING)
# ==============================================================================
orden_iir = 4             # Orden para los filtros IIR Butterworth
numtaps_fir = 101         # Número de coeficientes (taps) para los filtros FIR

# Frecuencias de corte (Hz)
fc_lpf = 50.0             # Corte pasa bajos: deja pasar 15 Hz, atenúa 120 y 300 Hz
fc_hpf = 220.0            # Corte pasa altos: deja pasar 300 Hz, atenúa 15 y 120 Hz
fc_bpf = [80.0, 160.0]    # Corte pasa bandas: deja pasar 120 Hz, atenúa 15 y 300 Hz

# --- A) Diseño IIR Butterworth ---
b_iir_lpf, a_iir_lpf = signal.butter(orden_iir, fc_lpf / nyq, btype='low')
b_iir_hpf, a_iir_hpf = signal.butter(orden_iir, fc_hpf / nyq, btype='high')
b_iir_bpf, a_iir_bpf = signal.butter(orden_iir, [fc_bpf[0] / nyq, fc_bpf[1] / nyq], btype='band')

# --- B) Diseño FIR con Ventana de Hamming ---
b_fir_lpf = signal.firwin(numtaps_fir, fc_lpf / nyq, window='hamming', pass_zero='lowpass')
b_fir_hpf = signal.firwin(numtaps_fir, fc_hpf / nyq, window='hamming', pass_zero='highpass')
b_fir_bpf = signal.firwin(numtaps_fir, [fc_bpf[0] / nyq, fc_bpf[1] / nyq], window='hamming', pass_zero='bandpass')


# ==============================================================================
# 3. APLICACIÓN DE LOS FILTROS A LA SEÑAL
# ==============================================================================
# Se utiliza filtfilt (filtrado de fase cero) para evitar desfase temporal al comparar
y_iir_lpf = signal.filtfilt(b_iir_lpf, a_iir_lpf, senal_ruidosa)
y_iir_hpf = signal.filtfilt(b_iir_hpf, a_iir_hpf, senal_ruidosa)
y_iir_bpf = signal.filtfilt(b_iir_bpf, a_iir_bpf, senal_ruidosa)

y_fir_lpf = signal.filtfilt(b_fir_lpf, [1.0], senal_ruidosa)
y_fir_hpf = signal.filtfilt(b_fir_hpf, [1.0], senal_ruidosa)
y_fir_bpf = signal.filtfilt(b_fir_bpf, [1.0], senal_ruidosa)


# ==============================================================================
# 4. VISUALIZACIÓN DE LOS RESULTADOS
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- GRÁFICA 1: Respuesta en Frecuencia de los Filtros (Bode de Magnitud) ---
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
config_filtros = [
    ("Pasa Bajos (fc = 50 Hz)", b_iir_lpf, a_iir_lpf, b_fir_lpf, [fc_lpf]),
    ("Pasa Bandas (fc = 80-160 Hz)", b_iir_bpf, a_iir_bpf, b_fir_bpf, fc_bpf),
    ("Pasa Altos (fc = 220 Hz)", b_iir_hpf, a_iir_hpf, b_fir_hpf, [fc_hpf])
]

for ax, (titulo, b_iir, a_iir, b_fir, cortes) in zip(axes1, config_filtros):
    w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=2048, fs=fs)
    w_fir, h_fir = signal.freqz(b_fir, [1.0], worN=2048, fs=fs)
    
    ax.plot(w_iir, 20 * np.log10(np.maximum(np.abs(h_iir), 1e-5)), label='IIR Butterworth (Orden 4)', lw=2)
    ax.plot(w_fir, 20 * np.log10(np.maximum(np.abs(h_fir), 1e-5)), label='FIR Hamming (101 taps)', lw=1.8, ls='--')
    for c in cortes:
        ax.axvline(c, color='crimson', linestyle=':', alpha=0.8, label=f'Corte: {c} Hz')
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylim(-80, 5)
    ax.legend(loc='lower left', fontsize=8.5)

axes1[0].set_ylabel('Magnitud (dB)')
fig1.suptitle('Respuesta en Frecuencia de los Filtros Digitales Diseñados', fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig('01_respuesta_frecuencia.png', dpi=300)

# --- GRÁFICA 2: Comparación en el Dominio del Tiempo (Zoom de 0 a 0.25 s) ---
ventana_t = t <= 0.25
fig2, axes2 = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axes2[0].plot(t[ventana_t], senal_ruidosa[ventana_t], color='gray', alpha=0.6, label='Señal con Ruido')
axes2[0].plot(t[ventana_t], senal_limpia[ventana_t], color='black', lw=1.5, label='Señal Compuesta Original (15+120+300 Hz)')
axes2[0].set_title('Señal de Entrada (Antes del Filtrado)', fontweight='bold')
axes2[0].legend(loc='upper right')

axes2[1].plot(t[ventana_t], np.sin(2 * np.pi * f_baja * t[ventana_t]), 'k:', alpha=0.7, label='Referencia 15 Hz')
axes2[1].plot(t[ventana_t], y_iir_lpf[ventana_t], color='#1f77b4', lw=1.8, label='Salida IIR Pasa Bajos')
axes2[1].plot(t[ventana_t], y_fir_lpf[ventana_t], color='#ff7f0e', lw=1.5, ls='--', label='Salida FIR Pasa Bajos')
axes2[1].set_title('Filtro Pasa Bajos (Recuperación de 15 Hz)', fontweight='bold')
axes2[1].legend(loc='upper right')

axes2[2].plot(t[ventana_t], 0.8 * np.sin(2 * np.pi * f_media * t[ventana_t]), 'k:', alpha=0.7, label='Referencia 120 Hz')
axes2[2].plot(t[ventana_t], y_iir_bpf[ventana_t], color='#2ca02c', lw=1.8, label='Salida IIR Pasa Bandas')
axes2[2].plot(t[ventana_t], y_fir_bpf[ventana_t], color='#d62728', lw=1.5, ls='--', label='Salida FIR Pasa Bandas')
axes2[2].set_title('Filtro Pasa Bandas (Recuperación de 120 Hz)', fontweight='bold')
axes2[2].legend(loc='upper right')

axes2[3].plot(t[ventana_t], 0.6 * np.sin(2 * np.pi * f_alta * t[ventana_t]), 'k:', alpha=0.7, label='Referencia 300 Hz')
axes2[3].plot(t[ventana_t], y_iir_hpf[ventana_t], color='#9467bd', lw=1.8, label='Salida IIR Pasa Altos')
axes2[3].plot(t[ventana_t], y_fir_hpf[ventana_t], color='#e377c2', lw=1.5, ls='--', label='Salida FIR Pasa Altos')
axes2[3].set_title('Filtro Pasa Altos (Recuperación de 300 Hz)', fontweight='bold')
axes2[3].set_xlabel('Tiempo (s)')
axes2[3].legend(loc='upper right')

for ax in axes2:
    ax.set_ylabel('Amplitud')
fig2.tight_layout()
fig2.savefig('02_comparativa_tiempo.png', dpi=300)

# --- GRÁFICA 3: Espectro de Frecuencias (FFT) Antes y Después ---
freqs = rfftfreq(len(t), 1 / fs)
calc_mag = lambda s: np.abs(rfft(s)) * 2 / len(t)

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
axes3[0, 0].plot(freqs, calc_mag(senal_ruidosa), color='dimgray')
axes3[0, 0].set_title('Espectro Original con Ruido (15, 120 y 300 Hz)', fontweight='bold')

axes3[0, 1].plot(freqs, calc_mag(y_iir_lpf), color='#1f77b4', label='IIR')
axes3[0, 1].plot(freqs, calc_mag(y_fir_lpf), color='#ff7f0e', ls='--', label='FIR')
axes3[0, 1].set_title('Espectro tras Filtro Pasa Bajos (15 Hz)', fontweight='bold')
axes3[0, 1].legend()

axes3[1, 0].plot(freqs, calc_mag(y_iir_bpf), color='#2ca02c', label='IIR')
axes3[1, 0].plot(freqs, calc_mag(y_fir_bpf), color='#d62728', ls='--', label='FIR')
axes3[1, 0].set_title('Espectro tras Filtro Pasa Bandas (120 Hz)', fontweight='bold')
axes3[1, 0].set_xlabel('Frecuencia (Hz)')
axes3[1, 0].legend()

axes3[1, 1].plot(freqs, calc_mag(y_iir_hpf), color='#9467bd', label='IIR')
axes3[1, 1].plot(freqs, calc_mag(y_fir_hpf), color='#e377c2', ls='--', label='FIR')
axes3[1, 1].set_title('Espectro tras Filtro Pasa Altos (300 Hz)', fontweight='bold')
axes3[1, 1].set_xlabel('Frecuencia (Hz)')
axes3[1, 1].legend()

for ax in axes3.flat:
    ax.set_ylabel('Magnitud')
    ax.set_xlim(0, 450)
fig3.suptitle('Evaluación en el Dominio de la Frecuencia (Transformada Rápida de Fourier)', fontsize=14, fontweight='bold')
fig3.tight_layout()
fig3.savefig('03_espectros_fft.png', dpi=300)

plt.show()