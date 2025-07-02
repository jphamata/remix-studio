# 🎹 Remix Studio: Signal Processing Toolkit

Welcome to **Remix Studio**, an all-in-one, browser-based toolkit for audio and image signal processing. This application, built with Gradio, provides a user-friendly interface for complex tasks like denoising, filtering, and audio source separation. It's designed for students, researchers, audio engineers, and hobbyists who need a powerful yet accessible tool for signal manipulation.

The application manages system resources, automatically detecting and utilizing a CUDA-enabled GPU for accelerated processing while providing robust fallbacks for CPU-only systems.

## ✨ Key Features

- **Dual-Mode Processing:** Seamlessly switch between processing **images** and **audio** files.
- **Noise Simulation:** Add Gaussian or Salt & Pepper (for images) noise to test the effectiveness of denoising algorithms.
- **Advanced Denoising Filters:**
  - **For Images:** Gaussian, Median, Wiener, and Bilateral filters.
  - **For Audio:** Spectrogram-based Gaussian and Median filtering to reduce noise while preserving signal characteristics.
- **Audio Source Separation:**
  - **🤖 AI-Powered (Deep Learning):** Utilizes the pre-trained **HDEMUCS** model from PyTorch to separate audio into `vocals`, `bass`, `drums`, and `other` stems.
  - **🎵 Classical (Morphological):** A fast and efficient method to separate an audio track into its `harmonic` and `percussive` components.
- **Rich Visualization:**
  - Real-time image previews (Original, Noisy, Processed).
  - Interactive audio players and detailed spectrogram plots.
- **Performance Metrics:**
  - **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) for image quality assessment.
  - **SDR** (Signal-to-Distortion Ratio) for audio quality assessment.
- **Intelligent Resource Management:**
  - Automatic CUDA (GPU) or CPU device detection.
  - Proactive memory monitoring and garbage collection to prevent crashes.
  - Chunk-based processing for the deep learning model to handle large audio files on memory-constrained systems.

## 🛠️ Getting Started

Follow these steps to set up and run Remix Studio on your local machine.

### Prerequisites

- Python 3
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jphamata/remix-studio.git
    cd remix-studio
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project's dependencies are listed in the `requirements.txt` file. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    > **💡 Note on PyTorch & CUDA:**
    > The `requirements.txt` file installs the CPU version of PyTorch for broad compatibility. If you have a CUDA-enabled NVIDIA GPU, you can achieve significant performance gains by installing a CUDA-specific version of PyTorch. Visit the official PyTorch website to find the correct command for your system and install it *before* running the command above.

## 🚀 Usage

Once the installation is complete, you can start the application.

### Running the Application

Execute the main Python script from your terminal:

```bash
python3 remix.py
```

After a few moments, you will see output in your terminal, including a local URL.

Open this URL in your web browser to access the Remix Studio interface.

### User Guide

The interface is designed to be intuitive and follows a simple workflow:

1.  **Step 1: Load Data & Add Noise**
    -   Upload an image or audio file using the respective input components. You can only work with one file type at a time.
    -   Configure the noise settings (type and intensity).
    -   Click **`🎯 Prepare Workspace & Add Noise`**. This will load your original file, create a noisy version, and display them in the visualizers.

2.  **Step 2: Denoise Your Signal**
    -   Navigate to the **`🛠 Denoising Tools`** tab.
    -   Select a denoising filter from the dropdown menu. The relevant parameters for that filter will appear below.
    -   Adjust the parameters as needed.
    -   Click **`🔧 Apply Denoise`**. The processed output will appear in the "Processed" visualizer, and performance metrics (PSNR, SSIM, or SDR) will be displayed.

3.  **Step 3: Separate Audio Sources (Audio Only)**
    -   If you have an audio file loaded, navigate to the **`🎛 Ferramentas de Separação de Fontes (Áudio)`** tab.
    -   **For Classical Separation:** Adjust the `Harmonic` and `Percussive` filter sliders and click `🎼 Separar Harmônico/Percussivo`.
    -   **For AI Separation:** Click `🚀 Separar com IA`. This process is resource-intensive and may take some time. A progress bar will indicate its status. The separated audio tracks will appear at the bottom.

4.  **Resetting:**
    -   At any point, you can click the **`🔄 Reset Workspace`** button at the top to clear all data, reset the interface, and free up system memory.

## 🔧 How It Works

-   **Audio Denoising:** The application performs a Short-Time Fourier Transform (STFT) to convert the audio signal into the frequency domain (a spectrogram). A 2D filter (Gaussian or Median) is applied to the magnitude of the spectrogram, and the signal is then reconstructed using the inverse STFT.
-   **Classical Separation:** This technique, known as morphological separation, leverages the assumption that harmonic components form horizontal lines in a spectrogram, while percussive components form vertical lines. Morphological filters are applied to isolate these structures.
-   **AI Separation:** This uses **Hybrid Demucs (HDEMUCS)**, a state-of-the-art deep neural network to identify and isolate different musical instruments.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   Project developed by **João Pedro Hamata**.
-   The Gradio team for making interactive machine learning apps accessible.
-   The developers of PyTorch, SciPy, and Scikit-image for their incredible open-source libraries.
