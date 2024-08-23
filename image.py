from huggingface_hub import hf_hub_download
import torch
from gguf_model import GGUFModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download a model from the Hugging Face Hub
    path = hf_hub_download(repo_id="city96/FLUX.1-dev-gguf", filename="flux1-dev-Q6_K.gguf")
    print(f"Model downloaded to {path}")

    # Initialisiere das Modell
    model = GGUFModel(path, device=device)

    # Lies den Prompt von der Konsole
    prompt = input("Enter a prompt: ")

    # Generiere ein Bild
    image = model.generate(prompt)

    # Hier könntest du das generierte Bild speichern oder anzeigen
    

if __name__ == "__main__":
    main()