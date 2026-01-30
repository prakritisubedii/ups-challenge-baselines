import torch
import webdataset as wds


def tar_url_for_number(tar_number: str, hf_token: str | None):
    tar_number = str(tar_number).zfill(6)
    if int(tar_number) <= 5000:
        base = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio"
    else:
        base = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio2"
    url = f"{base}/{tar_number}.tar?download=True"
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")
    token_header = f"Authorization:Bearer {hf_token}"
    return f"pipe:curl -s -L {url} -H {token_header}"


def fetch_mp3_bytes(tar_number: int, key: str, hf_token: str | None):
    url = tar_url_for_number(str(tar_number), hf_token)
    dataset = (
        wds.WebDataset([url], shardshuffle=False)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    )
    for mp3_bytes, sample_key, _ in dataset:
        if sample_key == key:
            return mp3_bytes
    return None


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None):
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(freq_hz):
        return 2595.0 * torch.log10(torch.tensor(1.0) + freq_hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    m_min = hz_to_mel(torch.tensor(f_min))
    m_max = hz_to_mel(torch.tensor(f_max))
    m_points = torch.linspace(m_min, m_max, n_mels + 2)
    hz_points = mel_to_hz(m_points)
    bin_freqs = torch.floor((n_fft + 1) * hz_points / sr).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left = bin_freqs[i].item()
        center = bin_freqs[i + 1].item()
        right = bin_freqs[i + 2].item()
        if center == left or right == center:
            continue
        for j in range(left, center):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / (right - center)
    return fb


def waveform_to_log_mel(
    waveform: torch.Tensor, sr: int, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80
):
    # waveform: [B, T]
    try:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )(waveform)
        log_mel = torch.log(mel + 1e-6)
        return log_mel
    except Exception:
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        power = stft.abs() ** 2
        fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(power.device)
        mel = torch.matmul(fb, power)
        log_mel = torch.log(mel + 1e-6)
        return log_mel
