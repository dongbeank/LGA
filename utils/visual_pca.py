import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

def visualize_pca(model, num_decoders, num_heads, head_dim, last_keys, n_vars=7):
    # Collect all q and k data for all n_vars
    all_q = []
    all_k = []
    for n_var in range(n_vars):
        for i in range(num_decoders):
            q = model.model.backbone.decoder.layers[i].cross_attn.q_s.detach().cpu()[n_var]
            k = model.model.backbone.decoder.layers[i].cross_attn.k_s.detach().cpu()[n_var]
            all_q.append(q)
            all_k.append(k)
    tau = 3
    # Reshape the data back to original structure
    q_length, k_length = all_q[0].shape[0], all_k[0].shape[0]
    
    # Perform PCA for each head separately
    reduced_q = []
    reduced_k = []
    max_abs_values = []  # Store max absolute values for each head
    for n_var in range(n_vars):
        reduced_q_var = []
        reduced_k_var = []
        max_abs_var = []
        for h in range(num_heads):
            q_data = torch.cat([q[:, h, :] for q in all_q[n_var*num_decoders:(n_var+1)*num_decoders]], dim=0)
            k_data = torch.cat([k[:, h, :] for k in all_k[n_var*num_decoders:(n_var+1)*num_decoders]], dim=0)
            
            pca = PCA(n_components=2)
            q_reduced = pca.fit_transform(q_data.numpy())
            k_reduced = pca.transform(k_data.numpy())
            
            reduced_q_var.append(q_reduced.reshape(num_decoders, q_length, 2))
            reduced_k_var.append(k_reduced.reshape(num_decoders, k_length, 2))
            
            # Calculate max absolute value for this head
            max_abs = max(np.abs(q_reduced).max(), np.abs(k_reduced).max())
            max_abs_var.append(max_abs)
        
        reduced_q.append(np.array(reduced_q_var).transpose(1, 2, 0, 3))  # Shape: (num_decoders, q_length, num_heads, 2)
        reduced_k.append(np.array(reduced_k_var).transpose(1, 2, 0, 3))  # Shape: (num_decoders, k_length, num_heads, 2)
        max_abs_values.append(max_abs_var)

    # Calculate the total number of rows
    total_rows = n_vars * num_decoders * num_heads
    
    # Create the figure with fixed width and calculated height
    plt.figure(figsize=(15, 3 * total_rows + 0.85))

    for n_var in range(n_vars):
        for i in range(num_decoders):
            for h in range(num_heads):
                for l in range(tau+1):
                    # Calculate the row index
                    row_idx = n_var * (num_decoders * num_heads) + i * num_heads + h
                    
                    ax = plt.subplot(total_rows, tau+1, (tau+1) * row_idx + l + 1)
                    
                    if l < tau:
                        k_data = reduced_k[n_var][i, :-last_keys, h][l::tau]
                        q_data = reduced_q[n_var][i, :, h][l::tau]
                        plt.scatter(k_data[:, 0], k_data[:, 1], label='K', alpha=0.5)
                        plt.scatter(q_data[0, 0], q_data[0, 1], label='Q1', alpha=0.5)
                        plt.scatter(q_data[1, 0], q_data[1, 1], label='Q2', alpha=0.5)
                        plt.scatter(reduced_k[n_var][i, -last_keys, h, 0], reduced_k[n_var][i, -last_keys, h, 1], label='Padding K', alpha=0.5, c='r')
                        plt.scatter(reduced_k[n_var][i, -last_keys+1:, h, 0], reduced_k[n_var][i, -last_keys+1:, h, 1], label='Learning K', alpha=0.5, c='k')
                        plt.title(f'n_var={n_var}, Decoder {i+1}, Head {h+1}, l={l}', fontsize=10)
                    else:
                        k_data = reduced_k[n_var][i, :, h]
                        q_data = reduced_q[n_var][i, :, h]
                        plt.scatter(k_data[:-last_keys, 0], k_data[:-last_keys, 1], label='K', alpha=0.5)
                        plt.scatter(q_data[:, 0], q_data[:, 1], label='Q', alpha=0.5)
                        plt.scatter(k_data[-last_keys, 0], k_data[-last_keys, 1], label='Padding K', alpha=0.5, c='r')
                        plt.scatter(k_data[-last_keys+1:, 0], k_data[-last_keys+1:, 1], label='Learning K', alpha=0.5, c='k')
                        plt.title(f'n_var={n_var}, Decoder {i+1}, Head {h+1}, All Data', fontsize=10)
                    
                    plt.xlabel('First Principal Component', fontsize=8)
                    plt.ylabel('Second Principal Component', fontsize=8)
                    
                    # Set axis limits based on max absolute value for this head
                    max_abs = max_abs_values[n_var][h]
                    ax.set_xlim(-max_abs, max_abs)
                    ax.set_ylim(-max_abs, max_abs)
                    
                    ax.set_aspect('equal', 'box')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.legend(loc='upper right', fontsize=6)

    plt.suptitle(f'PCA of K and Q Tensors for {n_vars} n_vars, {num_decoders} Decoders, {num_heads} Heads, and Data Subsets', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=1.5)
    plt.show()