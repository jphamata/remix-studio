# ==============================================================================
# PROJETO: 
#  - Ferramenta de Processamento de Sinal
# ==============================================================================
#
# AUTOR: João Pedro Hamata - 13672001
# DATA: 02/07/2025
#
# ==============================================================================

# Imports de Bibliotecas
import gradio as gr
import numpy as np
import torch
import torchaudio
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_bilateral
import soundfile as sf
import warnings
import gc
import psutil
import threading
import time
from contextlib import contextmanager
import traceback
from functools import wraps

# --- Configurações Globais ---
warnings.filterwarnings("ignore")

# Configurações de memória e processamento
MAX_AUDIO_DURATION = 180  # segundos
MAX_IMAGE_SIZE = 2048  # pixels máximos por dimensão
MAX_MEMORY_USAGE = 85  # % máximo de uso de memória
CHUNK_SIZE = 44100 * 10  # 10 segundos de áudio por chunk

# Device detection com fallback robusto
def get_device():
    try:
        if torch.cuda.is_available():
            # Testa se CUDA está realmente funcionando
            test_tensor = torch.randn(10, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            return torch.device("cuda:0")
    except:
        pass
    return torch.device("cpu")

DEVICE = get_device()
print(f"Usando dispositivo: {DEVICE}")
plt.style.use('seaborn-v0_8-darkgrid')

# ==============================================================================
# 1. UTILITÁRIOS DE GERENCIAMENTO DE RECURSOS
# ==============================================================================

def check_memory_usage():
    """Verifica o uso atual de memória do sistema."""
    return psutil.virtual_memory().percent

def clear_gpu_memory():
    """Limpa a memória da GPU se disponível."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_memory():
    """Força limpeza de memória."""
    gc.collect()
    clear_gpu_memory()

@contextmanager
def memory_monitor():
    """Context manager para monitorar uso de memória."""
    initial_memory = check_memory_usage()
    try:
        yield
    finally:
        current_memory = check_memory_usage()
        if current_memory > MAX_MEMORY_USAGE:
            clear_memory()
            print(f"Memória liberada: {initial_memory}% -> {check_memory_usage()}%")

def safe_execution(func):
    """Decorator para execução segura com tratamento de erros."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            with memory_monitor():
                return func(*args, **kwargs)
        except MemoryError:
            clear_memory()
            raise gr.Error("Memória insuficiente. Tente com arquivos menores ou use a função Reset.")
        except Exception as e:
            error_msg = f"Erro: {str(e)}"
            print(f"Erro detalhado: {traceback.format_exc()}")
            raise gr.Error(error_msg)
    return wrapper

def validate_audio(audio_data):
    """Valida e normaliza dados de áudio."""
    if not isinstance(audio_data, tuple) or len(audio_data) != 2:
        raise ValueError("Formato de áudio inválido")
    
    sr, audio = audio_data
    if sr <= 0:
        raise ValueError("Sample rate inválido")
    
    # Converte para numpy se necessário
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    audio = np.asarray(audio, dtype=np.float32)
    
    # Verifica duração
    duration = len(audio) / sr
    if duration > MAX_AUDIO_DURATION:
        raise ValueError(f"Áudio muito longo ({duration:.1f}s). Máximo: {MAX_AUDIO_DURATION}s")
    
    # Normaliza amplitude
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    return (sr, audio)

def validate_image(image_data):
    """Valida e redimensiona imagem se necessário."""
    if image_data is None:
        raise ValueError("Imagem inválida")
    
    img = np.asarray(image_data)
    
    # Verifica dimensões
    if img.ndim not in [2, 3]:
        raise ValueError("Imagem deve ter 2 ou 3 dimensões")
    
    # Redimensiona se muito grande
    if img.shape[0] > MAX_IMAGE_SIZE or img.shape[1] > MAX_IMAGE_SIZE:
        from skimage.transform import resize
        scale = MAX_IMAGE_SIZE / max(img.shape[:2])
        new_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))
        if img.ndim == 3:
            new_shape += (img.shape[2],)
        img = resize(img, new_shape, preserve_range=True, anti_aliasing=True)
        img = img.astype(np.uint8)
        print(f"Imagem redimensionada para {new_shape[:2]}")
    
    return img

# ==============================================================================
# 2. WORKSPACE E FUNÇÕES DE PROCESSAMENTO
# ==============================================================================

def create_initial_workspace():
    """Cria um dicionário de estado inicial para o workspace."""
    return {
        "original": None, 
        "noisy": None, 
        "processed": None, 
        "separated": {},
        "metadata": {"type": None, "size": None, "duration": None}
    }

@safe_execution
def reset_workspace():
    """Reseta completamente o workspace e libera memória."""
    clear_memory()
    return (
        create_initial_workspace(),
        # Limpa todas as visualizações
        None, None, None,  # imagens
        None, None, None,  # áudios
        None, None, None,  # espectrogramas
        "",  # métricas
        *[gr.update(value=None, visible=False) for _ in range(4)],  # separação
        gr.update(value="🔄 Workspace resetado com sucesso!")
    )

def sdr_numpy(reference, estimate, epsilon=1e-8):
    """Calcula SDR de forma mais robusta."""
    try:
        ref, est = np.asarray(reference, dtype=np.float32), np.asarray(estimate, dtype=np.float32)
        
        # Converte para mono se necessário
        if ref.ndim > 1: 
            ref = np.mean(ref, axis=1 if ref.shape[1] < ref.shape[0] else 0)
        if est.ndim > 1: 
            est = np.mean(est, axis=1 if est.shape[1] < est.shape[0] else 0)
        
        # Sincroniza comprimentos
        min_len = min(ref.shape[0], est.shape[0])
        ref, est = ref[:min_len], est[:min_len]
        
        if min_len == 0:
            return -np.inf
        
        # Calcula SDR
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + epsilon)
        s_target = alpha * ref
        e_noise = est - s_target
        
        pow_s_target = np.sum(s_target**2)
        pow_e_noise = np.sum(e_noise**2)
        
        if pow_s_target == 0:
            return -np.inf
        
        sdr_val = 10 * np.log10(pow_s_target / (pow_e_noise + epsilon) + epsilon)
        return np.clip(sdr_val, -50, 50)  # Limita valores extremos
    except:
        return -np.inf

@safe_execution
def plot_spectrogram(audio, sr, title="Espectrograma"):
    """Plota espectrograma de forma mais eficiente."""
    if audio is None: 
        return None
    
    try:
        # Converte para mono e reduz tamanho se necessário
        audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        
        # Reduz resolução para áudios longos
        if len(audio_mono) > CHUNK_SIZE:
            step = len(audio_mono) // CHUNK_SIZE
            audio_mono = audio_mono[::step]
            sr = sr // step
        
        # Calcula STFT com parâmetros otimizados
        nperseg = min(2048, len(audio_mono) // 4)
        noverlap = nperseg // 4 * 3
        
        f, t, Zxx = scipy.signal.stft(audio_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        # Converte para dB com clipping
        db_spec = 20 * np.log10(np.abs(Zxx) + 1e-9)
        
        # Cria plot
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Limita range dinâmico
        vmin = np.max(db_spec) - 80
        vmax = np.max(db_spec)
        
        mesh = ax.pcolormesh(t, f, db_spec, shading='gouraud', cmap='magma', 
                           vmin=vmin, vmax=vmax)
        
        ax.set(yscale='log', ylim=(20, sr/2), 
               xlabel="Tempo (s)", ylabel="Frequência (Hz)", title=title)
        
        fig.colorbar(mesh, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"Erro no espectrograma: {e}")
        return None

@safe_execution
def add_noise_to_data(data, noise_type, intensity):
    """Adiciona ruído aos dados de forma mais robusta."""
    if data is None: 
        return None
    
    try:
        if isinstance(data, np.ndarray) and data.ndim == 3:  # Imagem
            data = validate_image(data)
            img = data.astype(np.float32)
            amount = intensity / 100.0
            
            if noise_type == "Gaussiano":
                noise_std = amount * 50
                noise = np.random.normal(0, noise_std, img.shape)
                noisy = img + noise
            else:  # Sal e Pimenta
                noisy = np.copy(img)
                s_vs_p = 0.5
                
                # Sal
                num_salt = int(amount * img.size * s_vs_p)
                coords = tuple(np.random.randint(0, i-1, num_salt) for i in img.shape)
                noisy[coords] = 255
                
                # Pimenta
                num_pepper = int(amount * img.size * (1. - s_vs_p))
                coords = tuple(np.random.randint(0, i-1, num_pepper) for i in img.shape)
                noisy[coords] = 0
            
            return np.clip(noisy, 0, 255).astype(np.uint8)
            
        elif isinstance(data, tuple) and len(data) == 2:  # Áudio
            sr, audio = validate_audio(data)
            amount = intensity / 100.0
            
            # Adiciona ruído gaussiano
            noise = np.random.normal(0, np.std(audio) * amount * 2, audio.shape)
            noisy_audio = audio + noise
            
            # Normaliza para evitar clipping
            if np.max(np.abs(noisy_audio)) > 1.0:
                noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.95
            
            return (sr, noisy_audio.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Erro ao adicionar ruído: {str(e)}")
    
    return data

@safe_execution
def run_denoise(data, f_type, k_size, g_sigma, w_noise, b_sigma_s, b_sigma_c, original_data):
    """Executa denoising de forma mais robusta."""
    if data is None: 
        raise gr.Error("Execute o Passo 1 para adicionar ruído antes de aplicar o denoising.")
    
    try:
        if isinstance(data, np.ndarray) and data.ndim == 3:  # Imagem
            img = data.copy()
            k = int(k_size) // 2 * 2 + 1  # Garante kernel ímpar
            
            if f_type == "Gaussiano":
                proc = scipy.ndimage.gaussian_filter(img, sigma=g_sigma).astype(np.uint8)
            elif f_type == "Mediana":
                proc = scipy.ndimage.median_filter(img, size=k).astype(np.uint8)
            elif f_type == "Wiener (Imagem)":
                # Aplica Wiener por canal
                proc_channels = []
                for i in range(3):
                    channel = scipy.signal.wiener(img[:,:,i], k, w_noise)
                    proc_channels.append(channel)
                proc = np.clip(np.stack(proc_channels, axis=2), 0, 255).astype(np.uint8)
            elif f_type == "Bilateral (Imagem)":
                proc = (denoise_bilateral(img, sigma_color=b_sigma_c, 
                                        sigma_spatial=b_sigma_s, channel_axis=-1) * 255).astype(np.uint8)
            
            # Calcula métricas
            psnr_val = psnr(original_data, proc)
            ssim_val = ssim(original_data, proc, channel_axis=-1)
            metrics = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"
            
            return proc, metrics
            
        elif isinstance(data, tuple) and len(data) == 2:  # Áudio
            sr, audio = validate_audio(data)
            k = int(k_size) // 2 * 2 + 1
            
            # Converte para mono para processamento
            audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
            
            # STFT com parâmetros adaptativos
            nperseg = min(2048, len(audio_mono) // 4)
            noverlap = nperseg // 4 * 3
            
            f, t, Zxx = scipy.signal.stft(audio_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
            mag, phase = np.abs(Zxx), np.angle(Zxx)
            
            # Aplica filtro na magnitude
            if f_type == "Gaussiano":
                mag_filt = scipy.ndimage.gaussian_filter(mag, sigma=g_sigma)
            else:
                mag_filt = scipy.ndimage.median_filter(mag, size=k)
            
            # Reconstrói sinal
            Zxx_recon = mag_filt * np.exp(1j * phase)
            _, audio_proc = scipy.signal.istft(Zxx_recon, fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            # Garante mesmo tamanho do original
            if len(audio_proc) != len(audio_mono):
                audio_proc = np.resize(audio_proc, len(audio_mono))
            
            # Calcula SDR
            original_sr, original_audio = original_data
            original_mono = np.mean(original_audio, axis=1) if original_audio.ndim > 1 else original_audio
            sdr_val = sdr_numpy(original_mono, audio_proc)
            
            metrics = f"SDR: {sdr_val:.2f} dB"
            
            return (sr, audio_proc.astype(np.float32)), metrics
            
    except Exception as e:
        raise gr.Error(f"Erro no denoising: {str(e)}")
    
    return None, "Tipo de dado não suportado."

# ==============================================================================
# 3. MÓDULOS DE SEPARAÇÃO DE FONTES
# ==============================================================================

@safe_execution
def run_classical_separation(audio_tuple, h_len, p_len):
    """Separação clássica com melhor tratamento de erros."""
    if not isinstance(audio_tuple, tuple): 
        raise gr.Error("Separação Clássica requer um input de áudio. Por favor, carregue um áudio.")
    
    try:
        sr, audio = validate_audio(audio_tuple)
        audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        
        # Parâmetros adaptativos para STFT
        nperseg = min(2048, len(audio_mono) // 4)
        noverlap = nperseg // 4 * 3
        
        f, t, Zxx = scipy.signal.stft(audio_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        
        # Filtros morfológicos
        h_len_int, p_len_int = max(1, int(h_len)), max(1, int(p_len))
        
        mag_h = scipy.ndimage.grey_opening(mag, size=(1, h_len_int))
        mag_p = scipy.ndimage.grey_opening(mag, size=(p_len_int, 1))
        
        # Máscaras de separação
        mask_h = mag_h / (mag_h + mag_p + 1e-8)
        mask_p = mag_p / (mag_h + mag_p + 1e-8)
        
        # Reconstrói sinais
        _, audio_h = scipy.signal.istft(Zxx * mask_h, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, audio_p = scipy.signal.istft(Zxx * mask_p, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        return (
            gr.update(value=(sr, audio_h.astype(np.float32)), label="Harmônico", visible=True),
            gr.update(value=(sr, audio_p.astype(np.float32)), label="Percussivo", visible=True),
            gr.update(visible=False), 
            gr.update(visible=False)
        )
    except Exception as e:
        raise gr.Error(f"Erro na separação clássica: {str(e)}")

# Cache global para modelo
hdemucs_bundle = None
hdemucs_model = None

def get_hdemucs_bundle():
    """Carrega bundle HDEMUCS com cache."""
    global hdemucs_bundle
    if hdemucs_bundle is None:
        try:
            hdemucs_bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        except Exception as e:
            raise gr.Error(f"Erro ao carregar modelo HDEMUCS: {str(e)}")
    return hdemucs_bundle

def get_hdemucs_model():
    """Carrega modelo HDEMUCS com cache."""
    global hdemucs_model
    if hdemucs_model is None:
        try:
            bundle = get_hdemucs_bundle()
            hdemucs_model = bundle.get_model().to(DEVICE)
            hdemucs_model.eval()  # Modo de avaliação
        except Exception as e:
            clear_gpu_memory()
            raise gr.Error(f"Erro ao carregar modelo: {str(e)}")
    return hdemucs_model

def separate_sources_dl(model, mix, model_sr, seg=10.0, ov=0.1, dev=DEVICE):
    """Separação de fontes com processamento em chunks para economizar memória."""
    try:
        b, c, l = mix.shape
        chunk_l = int(model_sr * seg * (1 + ov))
        start, end = 0, chunk_l
        ov_f = int(ov * model_sr)
        
        fade = torchaudio.transforms.Fade(0, ov_f, "linear")
        final = torch.zeros(b, len(model.sources), c, l, device=dev)
        
        while start < l - ov_f:
            chunk = mix[:, :, start:end]
            
            # Processa chunk com limpeza de memória
            with torch.no_grad():
                out = model.forward(chunk.to(dev))
                if start > 0:  # Limpa chunks anteriores da GPU
                    del chunk
                    clear_gpu_memory()
            
            out = fade(out)
            final[:, :, :, start:end] += out.cpu()  # Move para CPU
            del out
            
            if start == 0:
                fade.fade_in_len = ov_f
            
            start += chunk_l - ov_f
            end += chunk_l - ov_f
            
            if end >= l:
                fade.fade_out_len = 0
        
        return final
    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        raise gr.Error("Memória GPU insuficiente. Tente com áudio mais curto.")
    except Exception as e:
        raise gr.Error(f"Erro na separação: {str(e)}")

@safe_execution
def run_dl_separation(audio_tuple, progress=gr.Progress()):
    """Separação deep learning com melhor gestão de recursos."""
    if not isinstance(audio_tuple, tuple): 
        raise gr.Error("Separação Deep Learning requer um input de áudio. Por favor, carregue um áudio.")
    
    try:
        progress(0, desc="Carregando modelo...")
        
        # Carrega modelo e configurações
        bundle = get_hdemucs_bundle()
        model = get_hdemucs_model()
        model_sr = bundle.sample_rate
        
        progress(0.1, desc="Preparando áudio...")
        
        sr, audio = validate_audio(audio_tuple)
        wave = torch.from_numpy(audio.T.astype(np.float32))
        
        # Reamostra se necessário
        if sr != model_sr:
            wave = torchaudio.functional.resample(wave, sr, model_sr)
        
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        
        # Normalização robusta
        ref = wave.mean(0)
        wave_std = ref.std()
        if wave_std > 0:
            wave = (wave - ref.mean()) / wave_std
        
        progress(0.3, desc="Separando fontes...")
        
        # Executa separação
        sources = separate_sources_dl(model, wave[None], model_sr=model_sr)[0]
        
        progress(0.9, desc="Finalizando...")
        
        # Desnormaliza
        if wave_std > 0:
            sources = sources * wave_std + ref.mean()
        
        # Converte para numpy e CPU
        results = []
        for s, n in zip(sources, model.sources):
            audio_np = s.cpu().numpy().T
            results.append(gr.update(value=(model_sr, audio_np), label=n.capitalize(), visible=True))
        
        # Preenche slots vazios
        while len(results) < 4:
            results.append(gr.update(visible=False))
        
        progress(1, desc="Concluído!")
        clear_memory()  # Limpa memória ao final
        
        return results
        
    except Exception as e:
        clear_memory()
        raise gr.Error(f"Erro na separação DL: {str(e)}")

# ==============================================================================
# 4. CONSTRUÇÃO DA INTERFACE GRÁFICA (UI/UX)
# ==============================================================================

with gr.Blocks(theme=gr.themes.Monochrome(), title="Remix Studio") as demo:
    ws = gr.State(create_initial_workspace)
    
    # Cabeçalho com informações do sistema
    with gr.Row():
        gr.Markdown("# 🎹 Remix Studio: Ferramenta de Processamento de Sinal (v2.0)")
        with gr.Column(scale=1):
            system_info = gr.Markdown(f"**Sistema:** {DEVICE} | **Memória:** {check_memory_usage():.1f}%")
            reset_btn = gr.Button("🔄 Reset Workspace", variant="secondary", size="sm")
    
    # Status e alertas
    status_message = gr.Markdown("✅ Sistema pronto. Carregue seus dados para começar.", visible=True)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Passo 1: Carregar Dados e Adicionar Ruído")
            
            input_img = gr.Image(
                type="numpy",
                label="Input de Imagem"
            )
            gr.Markdown(f"Máximo: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} pixels")

            input_audio = gr.Audio(
                type="numpy",
                label="Input de Áudio",
                sources=["upload", "microphone"]
            )
            gr.Markdown(f"Máximo: {MAX_AUDIO_DURATION}s")
            
            with gr.Accordion("Configurações de Ruído", open=True):
                noise_type = gr.Radio(
                    ["Gaussiano", "Sal e Pimenta"], 
                    value="Gaussiano", 
                    label="Tipo (Imagem)"
                )
                noise_intensity = gr.Slider(
                    0, 100, 10, 
                    label="Intensidade (%)",
                    info="Intensidade do ruído a ser adicionado"
                )
            
            prepare_btn = gr.Button("🎯 Preparar Workspace & Adicionar Ruído", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Visualizadores")
            with gr.Tabs():
                with gr.TabItem("🖼 Imagem"):
                    with gr.Row():
                        img_view_orig = gr.Image(label="Original", interactive=False)
                        img_view_noisy = gr.Image(label="Com Ruído", interactive=False)
                        img_view_proc = gr.Image(label="Processada", interactive=False)
                        
                with gr.TabItem("🎤 Áudio"):
                    with gr.Accordion("Players de Áudio", open=False):
                        with gr.Row():
                            audio_view_orig = gr.Audio(label="Original", interactive=False)
                            audio_view_noisy = gr.Audio(label="Com Ruído", interactive=False)
                            audio_view_proc = gr.Audio(label="Processado", interactive=False)
                    with gr.Row():
                        spec_view_orig = gr.Plot(label="Espectrograma Original")
                        spec_view_noisy = gr.Plot(label="Espectrograma Com Ruído")  
                        spec_view_proc = gr.Plot(label="Espectrograma Processado")
    
    gr.Markdown("### Passo 2: Aplicar Ferramentas de Processamento")
    
    with gr.Tabs():
        with gr.TabItem("🛠 Denoising Tools"):
            with gr.Row():
                with gr.Column():
                    denoise_filter_type = gr.Dropdown(
                        ["Gaussiano", "Mediana", "Wiener (Imagem)", "Bilateral (Imagem)"], 
                        value="Gaussiano", 
                        label="Filtro",
                        info="Escolha o tipo de filtro para denoising"
                    )
                    
                    # Parâmetros condicionais
                    with gr.Row(visible=True) as gaussian_params:
                        denoise_sigma_gauss = gr.Slider(0.1, 10.0, 1.0, label="Sigma (Intensidade)")
                    
                    with gr.Row(visible=False) as median_params:
                        denoise_kernel_size_median = gr.Slider(3, 21, 5, step=2, label="Tamanho do Kernel")
                    
                    with gr.Row(visible=False) as wiener_params:
                        denoise_kernel_size_wiener = gr.Slider(3, 21, 5, step=2, label="Tamanho do Kernel")
                        denoise_noise_wiener = gr.Slider(0.1, 1000, 10.0, label="Variância do Ruído")
                    
                    with gr.Row(visible=False) as bilateral_params:
                        denoise_sigma_s = gr.Slider(1, 30, 15, label="Sigma Espacial")
                        denoise_sigma_c = gr.Slider(0.01, 0.5, 0.1, label="Sigma Cor/Intensidade")
                    
                    run_denoise_btn = gr.Button("🔧 Aplicar Denoise", variant="primary")
                    
                with gr.Column():
                    metrics_label = gr.Label(label="Métricas de Desempenho", value="")

        with gr.TabItem("🎛 Ferramentas de Separação de Fontes (Áudio)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎵 Separação Clássica (Morfológica)")
                    gr.Markdown("*Rápida e eficiente para separação harmônico/percussivo*")
                    
                    h_len = gr.Slider(1, 100, 30, label="Filtro Harmônico", 
                                    info="Comprimento do filtro para componentes harmônicos")
                    p_len = gr.Slider(1, 100, 30, label="Filtro Percussivo",
                                    info="Comprimento do filtro para componentes percussivos")
                    run_classical_sep_btn = gr.Button("🎼 Separar Harmônico/Percussivo")
                    
                with gr.Column():
                    gr.Markdown("#### 🤖 Separação Deep Learning (Pré-treinado)")
                    gr.Markdown("*Separação avançada com IA - requer mais recursos*")
                    
                    gr.Markdown("**Fontes separadas:** Vocais, Baixo, Bateria, Outros")
                    memory_warning = gr.Markdown(
                        "⚠️ **Atenção:** Processo intensivo. Áudios longos podem usar muita memória.",
                        visible=True
                    )
                    run_dl_sep_btn = gr.Button("🚀 Separar com IA", variant="primary")
            
            gr.Markdown("#### 🎶 Saídas da Separação")
            with gr.Row():
                separated_outputs_audio = [
                    gr.Audio(label=f"Faixa {i+1}", visible=False, interactive=False) 
                    for i in range(4)
                ]

    # ==============================================================================
    # 5. LÓGICA DE EVENTOS DA UI
    # ==============================================================================
    
    # Eventos de reset e limpeza
    def update_system_info():
        """Atualiza informações do sistema."""
        memory_usage = check_memory_usage()
        color = "🟢" if memory_usage < 70 else "🟡" if memory_usage < 85 else "🔴"
        return f"**Sistema:** {DEVICE} | **Memória:** {color} {memory_usage:.1f}%"
    
    # Reset workspace
    reset_btn.click(
        reset_workspace,
        outputs=[
            ws, img_view_orig, img_view_noisy, img_view_proc,
            audio_view_orig, audio_view_noisy, audio_view_proc,
            spec_view_orig, spec_view_noisy, spec_view_proc,
            metrics_label, *separated_outputs_audio, status_message
        ]
    )
    
    # Mutual exclusion entre inputs
    def clear_audio_input():
        return gr.update(value=None)
    
    def clear_image_input():
        return gr.update(value=None)
    
    input_img.change(clear_audio_input, outputs=input_audio)
    input_audio.change(clear_image_input, outputs=input_img)
    
    # Atualização periódica do sistema (a cada 10 segundos)
    def periodic_update():
        return update_system_info()
    
    update_btn = gr.Button("Atualizar sistema")
    update_btn.click(periodic_update, outputs=system_info)
    
    # Preparação do workspace
    @safe_execution
    def prepare_workspace_and_add_noise(ws_dict, img, audio, noise_t, noise_i):
        """Prepara workspace e adiciona ruído com validação robusta."""
        # Limpa workspace anterior
        ws_dict = create_initial_workspace()
        
        # Determina tipo de dado
        data = img if img is not None else audio
        if data is None: 
            raise gr.Error("Por favor, carregue uma imagem ou áudio primeiro.")
        
        # Valida e armazena dados originais
        if isinstance(data, np.ndarray):  # Imagem
            data = validate_image(data)
            ws_dict["metadata"] = {
                "type": "image", 
                "size": data.shape[:2], 
                "duration": None
            }
        else:  # Áudio
            data = validate_audio(data)
            sr, audio_data = data
            ws_dict["metadata"] = {
                "type": "audio", 
                "size": audio_data.shape, 
                "duration": len(audio_data) / sr
            }
        
        ws_dict["original"] = data
        
        # Adiciona ruído
        noisy_data = add_noise_to_data(data, noise_t, noise_i)
        ws_dict["noisy"] = noisy_data
        
        # Prepara saídas baseadas no tipo
        if isinstance(data, np.ndarray):  # Imagem
            status = f"✅ Imagem carregada: {data.shape[0]}x{data.shape[1]} pixels"
            return (
                ws_dict, data, noisy_data, None, None, None, 
                None, None, None, status
            )
        else:  # Áudio
            sr, orig_audio = data
            sr_n, noisy_audio = noisy_data
            
            # Gera espectrogramas
            plot_orig = plot_spectrogram(orig_audio, sr, "Original")
            plot_noisy = plot_spectrogram(noisy_audio, sr_n, "Com Ruído")
            
            duration = len(orig_audio) / sr
            status = f"✅ Áudio carregado: {duration:.1f}s, {sr}Hz"
            
            return (
                ws_dict, None, None, data, noisy_data, None,
                plot_orig, plot_noisy, None, status
            )
    
    prepare_btn.click(
        prepare_workspace_and_add_noise,
        inputs=[ws, input_img, input_audio, noise_type, noise_intensity],
        outputs=[
            ws, img_view_orig, img_view_noisy, audio_view_orig, 
            audio_view_noisy, audio_view_proc, spec_view_orig, 
            spec_view_noisy, spec_view_proc, status_message
        ]
    )
    
    # Processamento de denoising
    @safe_execution
    def process_denoise_click(ws_dict, f_type, s_g, k_m, k_w, n_w, s_s, s_c):
        """Processa denoising com tratamento de erros."""
        if ws_dict["noisy"] is None:
            raise gr.Error("Nenhum dado com ruído encontrado. Execute o Passo 1 primeiro.")
        
        # Determina parâmetros baseados no tipo de filtro
        k_size = k_m if f_type == "Mediana" else (k_w if f_type == "Wiener (Imagem)" else 3)
        
        # Executa denoising
        processed_data, metrics = run_denoise(
            ws_dict["noisy"], f_type, k_size, s_g, n_w, s_s, s_c, ws_dict["original"]
        )
        
        ws_dict["processed"] = processed_data
        
        # Prepara saídas
        if isinstance(processed_data, np.ndarray):  # Imagem
            status = "✅ Denoising de imagem concluído"
            return ws_dict, processed_data, None, None, metrics, status
        else:  # Áudio
            plot_proc = plot_spectrogram(processed_data[1], processed_data[0], "Processado")
            status = "✅ Denoising de áudio concluído"
            return ws_dict, None, processed_data, plot_proc, metrics, status
    
    denoise_inputs = [
        ws, denoise_filter_type, denoise_sigma_gauss, denoise_kernel_size_median,
        denoise_kernel_size_wiener, denoise_noise_wiener, denoise_sigma_s, denoise_sigma_c
    ]
    
    run_denoise_btn.click(
        process_denoise_click,
        inputs=denoise_inputs,
        outputs=[ws, img_view_proc, audio_view_proc, spec_view_proc, metrics_label, status_message]
    )
    
    # Toggle de parâmetros de denoising
    def toggle_denoise_params(f_type):
        """Mostra/oculta parâmetros baseados no tipo de filtro."""
        return (
            gr.update(visible=f_type == "Gaussiano"),
            gr.update(visible=f_type == "Mediana"),
            gr.update(visible=f_type == "Wiener (Imagem)"),
            gr.update(visible=f_type == "Bilateral (Imagem)")
        )
    
    denoise_filter_type.change(
        toggle_denoise_params,
        inputs=denoise_filter_type,
        outputs=[gaussian_params, median_params, wiener_params, bilateral_params]
    )
    
    # Separação clássica
    @safe_execution
    def classical_separation_wrapper(ws_dict, h_len, p_len):
        """Wrapper para separação clássica."""
        # Determina áudio a processar
        audio_to_process = ws_dict.get('processed') or ws_dict.get('noisy') or ws_dict.get('original')
        
        if audio_to_process is None:
            raise gr.Error("Nenhum áudio carregado no workspace.")
        
        if not isinstance(audio_to_process, tuple):
            raise gr.Error("Separação clássica requer dados de áudio.")
        
        return run_classical_separation(audio_to_process, h_len, p_len)
    
    run_classical_sep_btn.click(
        classical_separation_wrapper,
        inputs=[ws, h_len, p_len],
        outputs=separated_outputs_audio
    )
    
    # Separação deep learning
    @safe_execution
    def dl_separation_wrapper(ws_dict, progress=gr.Progress()):
        """Wrapper para separação deep learning."""
        # Determina áudio a processar
        audio_to_process = ws_dict.get('processed') or ws_dict.get('noisy') or ws_dict.get('original')
        
        if audio_to_process is None:
            raise gr.Error("Nenhum áudio carregado no workspace.")
        
        if not isinstance(audio_to_process, tuple):
            raise gr.Error("Separação deep learning requer dados de áudio.")
        
        # Verifica duração para alertar sobre recursos
        sr, audio = audio_to_process
        duration = len(audio) / sr
        
        if duration > 60:
            progress(0, desc=f"⚠️ Áudio longo ({duration:.1f}s) - Isso pode demorar...")
        
        return run_dl_separation(audio_to_process, progress)
    
    run_dl_sep_btn.click(
        dl_separation_wrapper,
        inputs=[ws],
        outputs=separated_outputs_audio
    )
    
    # Validação de inputs em tempo real
    def validate_image_input(img):
        """Valida input de imagem em tempo real."""
        if img is None:
            return "📁 Carregue uma imagem"
        
        try:
            validated = validate_image(img)
            h, w = validated.shape[:2]
            size_mb = img.nbytes / (1024 * 1024)
            return f"✅ Imagem válida: {w}x{h} ({size_mb:.1f}MB)"
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    def validate_audio_input(audio):
        """Valida input de áudio em tempo real."""
        if audio is None:
            return "📁 Carregue um áudio"
        
        try:
            validated = validate_audio(audio)
            sr, audio_data = validated
            duration = len(audio_data) / sr
            size_mb = audio_data.nbytes / (1024 * 1024)
            return f"✅ Áudio válido: {duration:.1f}s, {sr}Hz ({size_mb:.1f}MB)"
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    # Eventos de validação em tempo real
    input_img.change(
        lambda img: validate_image_input(img),
        inputs=input_img,
        outputs=status_message
    )
    
    input_audio.change(
        lambda audio: validate_audio_input(audio),
        inputs=input_audio,
        outputs=status_message
    )

# ==============================================================================
# 6. CONFIGURAÇÕES DE INICIALIZAÇÃO E CLEANUP
# ==============================================================================

def cleanup_on_exit():
    """Limpa recursos ao sair."""
    global hdemucs_model, hdemucs_bundle
    
    try:
        if hdemucs_model is not None:
            del hdemucs_model
        if hdemucs_bundle is not None:
            del hdemucs_bundle
        clear_memory()
        print("🧹 Limpeza de recursos concluída")
    except:
        pass

import atexit
atexit.register(cleanup_on_exit)

# ==============================================================================
# 7. INICIALIZAÇÃO DO APLICATIVO
# ==============================================================================

if __name__ == "__main__":
    print("🎹 Iniciando Remix Studio v2.0...")
    print(f"📊 Sistema: {DEVICE}")
    print(f"💾 Memória inicial: {check_memory_usage():.1f}%")
    print(f"⚙️ Configurações:")
    print(f"   - Duração máxima de áudio: {MAX_AUDIO_DURATION}s")
    print(f"   - Tamanho máximo de imagem: {MAX_IMAGE_SIZE}px")
    print(f"   - Limite de memória: {MAX_MEMORY_USAGE}%")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            max_threads=4,  # Limita threads para controle de recursos
            favicon_path=None,
            ssl_verify=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Aplicativo interrompido pelo usuário")
        cleanup_on_exit()
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        cleanup_on_exit()
    finally:
        print("👋 Remix Studio finalizado")
