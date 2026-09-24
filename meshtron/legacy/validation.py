
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import math
import random
import pandas as pd
import seaborn as sns
import shutil


def load_data():
    base_path = "./checkpoints/"

    trained_model_path = os.listdir(base_path)

    history = []

    for trained_model in trained_model_path:

        if trained_model == 'saved_trainings':
            continue

        path = base_path + trained_model

        checkpoints = os.listdir(path)
        directories = [path + '/' + checkpoint for checkpoint in checkpoints]

        try:
            # checkpoints_path = path + '/' + chekpoints
            last_checkpoint = max(directories, key=os.path.getmtime)
        except:
            print(f"\n empty directory: {last_checkpoint}")
            try:
                shutil.rmtree(last_checkpoint)
            except Exception as e:
                print(f"Fehler beim Löschen von {last_checkpoint}: {e}")

            continue
        try:
            model_params = extract_params(last_checkpoint)

            saved_train_data = torch.load(last_checkpoint)
            history.append(
                [saved_train_data["training_history"], model_params])
        except:
            print(f'failed to load data of directory {last_checkpoint}')
    return history


def extract_params(dir_name):
    params = {}

    # 1. Einfache Integer-Parameter
    # Wir nutzen eine kleine Hilfsfunktion, um Redundanz zu vermeiden
    def get_int(pattern, default=None):
        match = re.search(pattern, dir_name)
        return int(match.group(1)) if match else default

    params['q'] = get_int(r'q_(\d+)')
    params['d_model'] = get_int(r'd_model_(\d+)')
    params['n_latents'] = get_int(r'n_latents_(\d+)')
    params['batch_size'] = get_int(r'batch_size_(\d+)')
    params['n_heads'] = get_int(r'n_heads_(\d+)')

    # 2. Sorting Strategy (Neu: Default 0 für alte Verzeichnisse)
    # Falls 'sorting_strategy_X' nicht gefunden wird, setzen wir 0
    sort_match = re.search(r'sorting_strategy_(\d+)', dir_name)
    params['sorting_strategy'] = int(sort_match.group(1)) if sort_match else 0

    # 3. Window size (kann None sein)
    window_match = re.search(r'window_size_(None|\d+)', dir_name)
    if window_match:
        val = window_match.group(1)
        params['window_size'] = None if val == 'None' else int(val)
    else:
        params['window_size'] = None

    # 4. Stage layers (Liste von Zahlen)
    # WICHTIG: Wir nutzen ([0-9_]+), damit wir nur Zahlen und Unterstriche matchen
    # und stoppen, sobald ein Buchstabe (der nächste Parameter-Name) kommt.
    stage_layers_match = re.search(r'stage_layers_([0-9_]+)', dir_name)
    if stage_layers_match:
        # Falls sorting_strategy dahinter steht, schneiden wir evtl. hängende _ ab
        layers_raw = stage_layers_match.group(1).strip('_')
        params['stage_layers'] = [
            int(x) for x in layers_raw.split('_') if x.isdigit()]

    return params


def plot_losses(history, output_file='./figures/validation_plots/loss_history_3D.png'):
    figsize = (15, 15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(history):
        color = (random.random(), random.random(), random.random())
        loss = data[0]
        params = data[1]
        train_loss = np.array(loss['train_loss'])
        val_loss = np.array(loss['val_loss'])
        epoch = np.array(loss['epoch'])

        z = np.ones(epoch.size) * 5*i  # Z-Wert für diesen Run

        ax.scatter(epoch, z, train_loss, s=5, marker='o', c=[color], zorder=10)
        ax.scatter(epoch, z, val_loss, s=5, marker='x', c=[color], zorder=10)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Run Number')
    ax.set_zlabel('Loss')
    ax.view_init(elev=20, azim=45)  # Viewing angle anpassen

    plt.savefig(output_file, dpi=600, transparent=True)


plot_losses(train_history)


def plot_loss_individual(history, output_dir='./figures/validation_plots'):
    # Sicherstellen, dass das Verzeichnis existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Wir nehmen die sorting_strategy mit in die Liste auf
    params_to_check = ['q', 'd_model', 'n_latents', 'n_heads',
                       'window_size', 'stage_layers', 'sorting_strategy']

    for param_name in params_to_check:
        # 1. Neues Figure-Objekt für diesen Parameter erstellen
        plt.figure(figsize=(10, 6))
        ax = plt.gca()  # Aktuelles Achsen-Objekt holen

        # --- Logik zur Ermittlung der einzigartigen Werte (wie gehabt) ---
        if param_name == 'stage_layers':
            unique_values = list(
                set(tuple(data[1].get(param_name, [])) for data in history))
            unique_values.sort()
        else:
            # .get(param_name, 0) sorgt dafür, dass alte Daten ohne sorting_strategy als 0 gewertet werden
            unique_values = list(set(data[1].get(
                param_name, 0 if param_name == 'sorting_strategy' else None) for data in history))
            unique_values.sort(key=lambda x: (x is None, x))

        # Farbmap erstellen
        # if len(unique_values) <= 10:
        #     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        # else:
        if len(unique_values) <= 10:
            colors = plt.cm.tab10(np.arange(len(unique_values)))
        else:
        colors = plt.cm.get_cmap('tab10')(
            np.linspace(0, 1, len(unique_values)))

        # color_map = {val: colors[i] for i, val in enumerate(unique_values)}

        # --- Alle Kurven für diesen Parameter plotten ---
        for data in history:
            loss = data[0]
            params = data[1]

            # Wert für diesen Parameter holen (mit Default-Handling für sorting_strategy)
            raw_val = params.get(param_name, 0 if param_name ==
                                 'sorting_strategy' else None)
            param_value = tuple(
                raw_val) if param_name == 'stage_layers' else raw_val

            train_loss = np.array(loss['train_loss'])
            val_loss = np.array(loss['val_loss'])
            epoch = np.array(loss['epoch'])

            color = color_map[param_value]

            ax.plot(epoch, train_loss, alpha=0.5, color=color,
                    label=f'{param_name}={param_value}')
            ax.plot(epoch, val_loss, alpha=0.5, linestyle='--', color=color)

        # --- Layout-Details ---
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  fontsize=8, loc='upper right', bbox_to_anchor=(1.15, 1))

        ax.set_title(f'Loss Analysis: {param_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # --- Speichern ---
        # Dateiname generieren: z.B. loss_by_sorting_strategy.png
        filename = f"loss_by_{param_name}.png"
        save_path = os.path.join(output_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # WICHTIG: Speicher freigeben, sonst hängen hunderte Plots im RAM
        plt.close()

    print(f"Alle Plots wurden in {output_dir} gespeichert.")


def plot_loss_by_params(history, output_file='./figures/validation_plots/loss_history_params.png'):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    params_to_check = ['q', 'd_model', 'n_latents',
                       'n_heads', 'window_size', 'stage_layers', 'sorting_strategy']
    n_params = len(params_to_check)

    if n_params % 2 < n_params % 3:
        n_cols = 2
    else:
        n_cols = 3

    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

# Flatten axes array für einfacheren Zugriff
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, param_name in enumerate(params_to_check):
        ax = axes[idx]

        # Sammle alle einzigartigen Parameterwerte
        if param_name == 'stage_layers':
            # Konvertiere Listen zu Tupeln für Hashbarkeit
            unique_values = list(
                set(tuple(data[1][param_name]) for data in history))
            # Sortiere Tupel
            unique_values.sort()
        else:
            unique_values = list(set(data[1][param_name] for data in history))
            # Sortiere mit None-Handling
            unique_values.sort(key=lambda x: (x is None, x))

        # Erstelle Farbmap mit besserer Unterscheidbarkeit
        if len(unique_values) <= 10:
            colors = plt.cm.tab10(np.arange(len(unique_values)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))

        color_map = {val: colors[i] for i, val in enumerate(unique_values)}

        for data in history:
            loss = data[0]
            params = data[1]

            train_loss = np.array(loss['train_loss'])
            val_loss = np.array(loss['val_loss'])
            epoch = np.array(loss['epoch'])

            # Konvertiere zu Tupel falls stage_layers
            if param_name == 'stage_layers':
                param_value = tuple(params[param_name])
            else:
                param_value = params[param_name]

            color = color_map[param_value]

            # Plot mit gleicher Farbe für gleichen Parameter
            ax.plot(epoch, train_loss, alpha=0.5, color=color,
                    label=f'{param_name}={param_value}')
            ax.plot(epoch, val_loss, alpha=0.5, linestyle='--', color=color)

        # Entferne doppelte Labels in der Legende
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        ax.set_title(f'Loss by {param_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.savefig(output_file, dpi=300)


def heatmap(history, output_file='./figures/validation_plots/loss_history_heatmap.png'):
    data_list = []
    for data in history:
        loss = data[0]
        params = data[1]

        try:
            row = params.copy()
            row['final_train_loss'] = loss['train_loss'][-1]
            row['final_val_loss'] = loss['val_loss'][-1]
            row['min_val_loss'] = min(loss['val_loss'])
            data_list.append(row)
        except:
            print('list error')
            continue
    df = pd.DataFrame(data_list)
    df['window_size'] = df['window_size'].fillna(0)  # None → 0

    param_cols = ['q', 'd_model', 'n_latents',
                  'batch_size', 'n_heads', 'window_size', 'stage_layers', 'sorting_strategy']
    loss_cols = ['final_train_loss', 'final_val_loss', 'min_val_loss']

    correlation = df[param_cols + loss_cols].corr().loc[param_cols, loss_cols]

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Parameter Influence on Loss')
    plt.savefig(output_file, dpi=300)


history = load_data()
plot_losses(history)
plot_loss_individual(train_history)
plot_loss_by_params(history)
heatmap(history)
