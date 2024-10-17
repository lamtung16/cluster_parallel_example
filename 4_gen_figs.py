# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv("final_result.csv").sort_values(by=['num_layers', 'layer_size', 'activation'])

# %%
for num_layer in sorted(df['num_layers'].unique()):
    # Filter the DataFrame
    df_filtered = df[(df['num_layers'] == num_layer)].copy()
    df_filtered = df_filtered.groupby(['num_layers', 'layer_size'])[['train_loss', 'val_loss']].mean().reset_index()

    # Ensure layer_size is compatible with int first, then convert to string
    df_filtered['layer_size'] = df_filtered['layer_size'].astype(int).astype(str)

    # Create a new figure
    plt.figure(figsize=(6, 3))

    # Plotting with categorical x-axis
    plt.plot(df_filtered['layer_size'], np.log(df_filtered['train_loss']), label='Train Loss', marker='o')
    plt.plot(df_filtered['layer_size'], np.log(df_filtered['val_loss']), label='Validation Loss', marker='o')

    # Add titles and labels
    plt.title(f'Train and Validation Loss vs Layer Size (num_layers = {num_layer})')
    plt.xlabel('Layer Size')
    plt.ylabel('Log Loss')
    plt.xticks(rotation=45)  # Rotate the x-axis labels if needed
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'figures/num_layers_{num_layer}_loss_plot.png', bbox_inches='tight')  # Save with a specific name
    plt.close()


