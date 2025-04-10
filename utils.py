from collections.abc import Iterable
from typing import cast
import torch
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.HookedTransformer import TransformerBlock  # type: ignore
from dataclasses import dataclass

def visualize_text_sequence_nmse(
    input_sequence: list[str],
    acts_base_SLD: np.ndarray,
    acts_tuned_SLD: np.ndarray,
    layer_names: list[str],
):
    """
    Visualizes the Normalized Mean Squared Error (NMSE) between base and tuned activations
    for each token across different layers.
    
    Args:
        input_sequence: List of input tokens to display
        acts_base_SLD: Base model activations with shape (sequence_length, layers, dimension)
        acts_tuned_SLD: Tuned model activations with shape (sequence_length, layers, dimension)
        layer_names: Names of the layers to display on the x-axis
    
    Returns:
        A plotly figure object displaying the NMSE heatmap
    """
    # Calculate NMSE between base and tuned activations
    nmse_SL = normalised_distance(acts_base_SLD, acts_tuned_SLD)
    
    # Get dimensions
    num_tokens = len(input_sequence)
    num_layers = nmse_SL.shape[1]
    
    # Create the heatmap
    fig = go.Figure(
        go.Heatmap(
            z=nmse_SL,
            y=input_sequence,
            x=layer_names,
            hovertemplate="Token: %{y}<br>Layer: %{x}<br>NMSE: %{z:.4f}<extra></extra>",
            showscale=True,
            ygap=0,
        )
    )
    
    # Update layout
    fig.update_layout(
        height=num_tokens * 10 + 100,  # Ensure minimum height with adaptive scaling
        width=num_layers * 20 + 150,  # Ensure minimum width with adaptive scaling
        title="Token-wise NMSE Between Base and Tuned Models",
        xaxis_title="Layer",
        yaxis_title="Token",
        margin=dict(t=80, b=20, l=20, r=20),
    )
    
    # Update axes
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=layer_names,
    )
    
    fig.update_yaxes(
        tickfont=dict(size=10),
        showgrid=False,
    )
    
    return fig

def visualise_text_sequence_vertical(
    input_sequence: list[str],
    base_pred_toks: list[str],
    tuned_pred_toks: list[str],
    mse_SL: np.ndarray,
    acts_base_SLD: np.ndarray,
    acts_tuned_SLD: np.ndarray,
    layer_names: list[str],
):
    assert len(base_pred_toks) == len(tuned_pred_toks)
    assert len(base_pred_toks) == len(input_sequence)

    # Calculate number of tokens and layers
    num_tokens = len(input_sequence)
    num_layers = mse_SL.shape[1]

    # Create figure with horizontal layout - just two heatmaps
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"type": "heatmap"}],
        ],
        horizontal_spacing=0.01,
    )

    # NMSE heatmap
    nmse_SL = normalised_distance(acts_base_SLD, acts_tuned_SLD)
    
    # We'll use reversed y indices, so no need to flip the data
    fig.add_trace(
        go.Heatmap(
            z=nmse_SL,  # Original data (will be displayed bottom-to-top)
            y=input_sequence,
            x=layer_names,
            hovertemplate="Token index: %{y}<br>Layer: %{x}<br>NMSE: %{z:.4f}<extra></extra>",
            showscale=False,
            ygap=0,  # Remove gaps between cells
        ),
        row=1,
        col=1,
    )

    # Update axes labels
    fig.update_xaxes(
        # showticklabels=False,  # Hide x-axis labels for KL (single column)
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title=None,
        row=1,
        col=1,
        tickmode="array",
        tickvals=list(range(num_tokens)),  # Reversed token indices
        ticktext=input_sequence,
        tickfont=dict(size=10),
        showgrid=False,
    )

    # For MSE plot
    fig.update_xaxes(
        row=1,
        col=1,
        title="Layer",
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=layer_names,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title="Token",
        showticklabels=True,
        showgrid=False,
    )

    # Update layout
    fig.update_layout(
        height=num_tokens * 40 + 50,  # Scale height based on number of tokens
        width=num_layers * 30 + 150,  # Scale width based on layers
        title="Token-wise NMSE",
        margin=dict(t=80, b=20, l=20, r=20),  # Increased left margin for token labels
    )

    # Return the figure for display
    return fig

def normalised_distance(acts_base_SLD: np.ndarray, acts_tuned_SLD: np.ndarray) -> np.ndarray:
    tuned_norms_SL = np.linalg.norm(acts_tuned_SLD, ord=2, axis=-1)
    base_norms_SL = np.linalg.norm(acts_base_SLD, ord=2, axis=-1)
    mean_l2_norm_SL = (tuned_norms_SL + base_norms_SL) / 2
    nmse_SL = np.linalg.norm(acts_tuned_SLD - acts_base_SLD, axis=-1) / mean_l2_norm_SL
    return nmse_SL

def visualise_text_sequence(
    input_sequence: list[str],
    base_pred_toks: list[str],
    tuned_pred_toks: list[str],
    kl_div_S: np.ndarray,
    mse_SL: np.ndarray,
):
    """
    Visualize a text sequence with corresponding KL divergence and MSE values.

    Args:
        input_sequence: List of input tokens/text segments
        base_pred_toks: Token predictions from math model
        tuned_pred_toks: Token predictions from r1 model
        kl_div_S: KL divergence values per token (shape: [S])
        mse_SL: MSE values per token and layer (shape: [S, L])
    """
    assert kl_div_S.shape[0] == mse_SL.shape[0]
    assert kl_div_S.shape[0] == len(input_sequence)
    assert len(base_pred_toks) == len(tuned_pred_toks)
    assert len(base_pred_toks) == len(input_sequence)

    # Calculate number of tokens and layers
    num_tokens = len(input_sequence)
    num_layers = mse_SL.shape[1]

    # Create figure with specified layout
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.2, 0.05, 0.75],  # Adjusted heights for better visibility
        specs=[
            [{"type": "table"}],  # For tokens (vertical)
            [{"type": "heatmap"}],  # For KL divergence
            [{"type": "heatmap"}],  # For MSE
        ],
        subplot_titles=(
            "Input Tokens",
            "KL Divergence by Token",
            "MSEs by Token and Layer (normed by layer)",
        ),
        vertical_spacing=0.03,
    )

    # Row 1: Create a vertical display of tokens
    # Transpose the input sequence to make it display vertically
    token_table = [[token, base_pred_tok, tuned_pred_tok] for token, base_pred_tok, tuned_pred_tok in zip(input_sequence, base_pred_toks, tuned_pred_toks)]

    fig.add_trace(
        go.Table(
            cells=dict(
                values=token_table,
                align="center",
                font=dict(size=10),
                height=25,
            ),
        ),
        row=1,
        col=1,
    )

    # Row 2: KL divergence heatmap
    # Create hover text with token information
    hover_texts = [
        f"Token: '{input_sequence[i]}' | Math: '{base_pred_toks[i]}' | R1: '{tuned_pred_toks[i]}' | KL: {kl_div_S[i]:.4f}"
        for i in range(num_tokens)
    ]

    fig.add_trace(
        go.Heatmap(
            z=[kl_div_S],  # Make it 2D for heatmap
            x=list(range(num_tokens)),
            y=[0],  # Single row
            coloraxis="coloraxis1",
            hoverinfo="text",
            text=[hover_texts],
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Transpose MSE matrix for better visualization (layers on y-axis)
    mse_LS = mse_SL.T
    max_per_layer_L = mse_LS.max(axis=1)
    mse_normed_by_layer_LS = mse_LS / max_per_layer_L[:, None]


    fig.add_trace(
        go.Heatmap(
            z=mse_normed_by_layer_LS,
            x=list(range(num_tokens)),  # Tokens on x-axis
            y=list(range(num_layers)),  # Layers on y-axis
            # coloraxis="coloraxis2",
            hovertemplate="Token index: %{x}<br>Layer: %{y}<br>MSE: %{z:.4f}<extra></extra>",
            # zmin=0,
            # zmax=10,
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(
        coloraxis1=dict(
            colorscale="Reds",
            colorbar=dict(
                title="KL Divergence",
                y=0.8,  # Position for KL colorbar
                len=0.2,
            ),
        ),
        height=800,  #max(800, num_layers * 20 + 400),  # Scale height based on layers
        width=num_tokens * 60,  # Scale width based on tokens
        title="Token-wise Analysis with KL Divergence and MSE",
        margin=dict(t=80, b=50, l=80, r=50),
    )

    # Ensure the plots align by adjusting axes
    # For KL divergence plot
    fig.update_xaxes(
        title="Token Index",
        range=[-0.5, num_tokens - 0.5],
        row=2,
        col=1,
        tickmode="array",
        tickvals=list(range(num_tokens)),
        ticktext=[f"{i}" for i in range(num_tokens)],
    )
    fig.update_yaxes(
        showticklabels=False,  # Hide y-axis labels for KL (single row)
        row=2,
        col=1,
    )

    # For MSE plot
    fig.update_xaxes(
        title="Token Index",
        range=[-0.5, num_tokens - 0.5],
        row=3,
        col=1,
        tickmode="array",
        tickvals=list(range(num_tokens)),
        ticktext=[f"{i}" for i in range(num_tokens)],
    )
    fig.update_yaxes(
        title="Layer",
        range=[-0.5, num_layers - 0.5],
        row=3,
        col=1,
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=[f"{i}" for i in range(num_layers)],
    )

    # Return the figure for display
    return fig

def _get_logits_and_resid( prompt: str, model: HookedTransformer, hookpoints: list[str] ):
    toks_1S: torch.Tensor = model.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert toks_1S.shape[0] == 1
    seq_logits, cache = model.run_with_cache(
        toks_1S,
        names_filter=lambda name: name in hookpoints,
    )
    toks_S = toks_1S[0]
    seq_logits_SV = seq_logits[0]
    cache_ = cache.remove_batch_dim()
    resid_SLD = torch.stack([cache_.cache_dict[hp] for hp in hookpoints ]).transpose(0, 1)
    return seq_logits_SV, resid_SLD, toks_S


def tokenwise_kl(probs_P_SV: torch.Tensor, probs_Q_SV: torch.Tensor):
    """S = seq, V = vocab"""
    tokens_kl_S = torch.sum(probs_P_SV * torch.log(probs_P_SV / probs_Q_SV), dim=-1)
    return tokens_kl_S



@dataclass
class SeqData:
    input_tokens_S: torch.Tensor
    base_pred_toks_S: torch.Tensor
    tuned_pred_toks_S: torch.Tensor
    kl_div_S: torch.Tensor
    acts_base_SLD: torch.Tensor
    acts_tuned_SLD: torch.Tensor


def run_acts_through_other_model(resid_mid_SLD: torch.Tensor, other_model: HookedTransformer) -> torch.Tensor:
    resid_post_SLD = torch.zeros_like(resid_mid_SLD)
    seq_len, n_layers, dim = resid_mid_SLD.shape
    for l, block in enumerate(cast(Iterable[TransformerBlock], other_model.blocks)):
        # block.hook_mlp_out
        mlp_out_1SD = block.apply_mlp(block.ln2(resid_mid_SLD[:, l, :][None]))
        assert mlp_out_1SD.shape == (1, seq_len, dim)
        resid_post_SLD[:, l, :] = mlp_out_1SD[0]
    return resid_post_SLD

def get_seq_data(prompt: str, llm_base: HookedTransformer, llm_tuned: HookedTransformer, hookpoints: list[str]) -> SeqData:
    base_logits_SV, base_resid_SLD, base_toks_S = _get_logits_and_resid(prompt, llm_base, hookpoints)
    tuned_logits_SV, tuned_resid_SLD, tuned_toks_S = _get_logits_and_resid(prompt, llm_tuned, hookpoints)

    assert (base_toks_S == tuned_toks_S).all()
    input_tokens_S = base_toks_S

    base_seq_probs_SV = base_logits_SV.softmax(dim=-1)
    tuned_seq_probs_SV = tuned_logits_SV.softmax(dim=-1)

    base_seq_preds_S = base_logits_SV.argmax(dim=-1)
    tuned_seq_preds_S = tuned_logits_SV.argmax(dim=-1)

    kl_div_S = tokenwise_kl(probs_P_SV=base_seq_probs_SV, probs_Q_SV=tuned_seq_probs_SV)

    return SeqData(
            input_tokens_S=input_tokens_S,
            base_pred_toks_S=base_seq_preds_S,
            tuned_pred_toks_S=tuned_seq_preds_S,
            kl_div_S=kl_div_S,
            acts_base_SLD=base_resid_SLD,
            acts_tuned_SLD=tuned_resid_SLD,
        )
