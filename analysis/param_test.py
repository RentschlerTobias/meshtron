
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from meshtron.training.trainer import Trainer
import torch


def main():

    path_meshes = '../data/structured_quad_meshes_pre_selected.pt'

    quantization = 2048
    d_model = 1024
    n_latents = 2048
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    stage_layers = [2, 2, 2, 2, 2]
    window_size = None
    gradient_accumulation = 2
    n_heads = 8
    verbose = False

    counter = 0
    counter += 1
    print(f'\n steup number {counter} \n')
    trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                      gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose)

    trainer.training()


if __name__ == "__main__":
    main()
