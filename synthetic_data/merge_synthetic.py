# synthetic_data/merge_synthetic.py
import pandas as pd
from .generate_faker import generate_faker_text
from .generate_gemini import generate_gemini_text

def add_synthetic_data(df_existing, method="faker", n_rows=1000, gemini_prompt=None, gemini_api_key=None):
    """
    method: "faker" veya "gemini"
    """
    if method == "faker":
        df_synth = generate_faker_text(n_rows=n_rows)
    elif method == "gemini":
        if gemini_prompt is None or gemini_api_key is None:
            raise ValueError("Gemini için prompt ve API key gereklidir.")
        df_synth = generate_gemini_text(gemini_prompt, gemini_api_key)
    else:
        raise ValueError("method sadece 'faker' veya 'gemini' olabilir")

    # Kolon eşleşmesi
    common_cols = [col for col in df_existing.columns if col in df_synth.columns]
    df_existing = df_existing.copy()
    df_synth = df_synth[common_cols]

    # Birleştir
    df_combined = pd.concat([df_existing, df_synth], ignore_index=True)
    return df_combined
