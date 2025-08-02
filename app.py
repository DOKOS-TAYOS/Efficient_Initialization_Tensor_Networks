import streamlit as st
import torch
import time
import io
from normalizer_module import MPS, MPO
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Tensor Network Initializer",
    page_icon="⚛️",
    layout="wide"
)

# Create two columns for the layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.title("Tensor Network Initializer 🔄")
    st.markdown("""
This application implements efficient initialization methods for tensor network layers using partial norm computations, 
as described in the paper [Efficient Finite Initialization with Partial Norms for Tensorized Neural Networks and Tensor Networks Algorithms](https://arxiv.org/abs/2309.06577) by Alejandro Mata Ali et al.

The method uses partial computations of Frobenius or linear entrywise norms in an iterative way to normalize tensor networks. 
Key features:
- 🚫 Prevents divergence and zero norm issues during initialization
- ⚡ Reuses intermediate calculations for efficiency
- 🔄 Supports both MPS (Matrix Product State) and MPO (Matrix Product Operator) layers
- 🎚️ Provides control over initialization parameters and tolerances
""")

    # Layer type selection
    layer_type = st.selectbox(
        "Select layer type 🔀",
        ["Matrix Product State", "Matrix Product Operator"],
        help="Choose between Matrix Product State (MPS) or Matrix Product Operator (MPO) layer"
    )

    # Normalization method
    norm_method = st.selectbox(
        "Select normalization method 📊",
        ["Frobenius", "Lineal"],
        help="Choose the normalization method to use"
    )

with right_col:
    st.subheader("Layer Parameters 🎛️")
    
    # Basic parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_nodes = st.number_input(
            "Number of nodes 🔢",
            min_value=2,
            value=30,
            help="Number of tensors in the network"
        )
        
    with col2:
        phys_dim = st.number_input(
            "Physical dimension 📏",
            min_value=1,
            value=12,
            help="Dimension of physical indices"
        )
        
    with col3:
        bond_dim = st.number_input(
            "Bond dimension 🔗",
            min_value=1,
            value=10,
            help="Dimension of bonds between tensors"
        )

    # Initialization parameters
    st.subheader("Initialization Parameters ⚙️")
    col1, col2, col3 = st.columns(3)
    with col1:
        mean = st.number_input(
            "Mean 📈",
            value=1.0,
            step=0.1,
            help="Mean for random initialization"
        )
        
    with col2:
        std = st.number_input(
            "Standard deviation 📊",
            value=0.5,
            step=0.1,
            help="Standard deviation for random initialization"
        )
        
    with col3:
        seed = st.number_input(
            "Random seed 🎲",
            value=None,
            step=1,
            help="Seed for random initialization (optional)"
        )

    # Tolerance parameters
    st.subheader("Tolerance Parameters 🎯")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_tol = st.number_input(
            "Maximum tolerance ⬆️",
            value=1e3,
            format="%.2e",
            help="Maximum tolerance for normalization"
        )
        
    with col2:
        min_tol = st.number_input(
            "Minimum tolerance ⬇️",
            value=1e-3,
            format="%.2e",
            help="Minimum tolerance for normalization"
        )

    with col3:
        use_default_norm = st.checkbox(
            "Use default norm factor",
            value=True,
            help="If checked, uses phys_dim^n_nodes as norm factor. Otherwise allows custom input."
        )
        if not use_default_norm:
            norm_factor = st.number_input(
                "Custom norm factor 📊",
                value=1.0,
                format="%.2e",
                help="Custom normalization factor"
            )
        else:
            norm_factor = phys_dim**n_nodes

    # Initialize button
    if st.button("Initialize Layer 🚀", type="primary"):
        with st.spinner("Initializing tensor network layer... ⏳"):
            try:
                # Set random seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                
                # Record start time
                start_time = time.time()
                
                # Initialize layer
                if layer_type == "MPS":
                    layer = MPS(
                        n_nodes=n_nodes,
                        phys_dim=phys_dim,
                        bond_dim=bond_dim,
                        std=std,
                        mean=mean,
                        max_tol=max_tol,
                        min_tol=min_tol,
                        method=norm_method,
                        norm_factor=phys_dim**(n_nodes)
                    )
                else:  # MPO
                    layer = MPO(
                        n_nodes=n_nodes,
                        phys_dim=phys_dim,
                        bond_dim=bond_dim,
                        std=std,
                        mean=mean,
                        max_tol=max_tol,
                        min_tol=min_tol,
                        method=norm_method,
                        norm_factor=phys_dim**(n_nodes)
                    )
                
                # Record end time
                end_time = time.time()
                runtime = end_time - start_time

                # Save tensors button
                tensors = [node.tensor for node in layer.layer_nodes]
                buffer = io.BytesIO()
                torch.save(tensors, buffer)
                st.download_button(
                    label="Download Tensors 💾", 
                    data=buffer.getvalue(),
                    file_name=f"{layer_type.lower()}_tensors.pt",
                    mime="application/octet-stream"
                )
                
                # Display results
                st.subheader("Results 📊")
                
                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of steps 🔄", layer.steps)
                with col2:
                    st.metric("Runtime (seconds) ⏱️", f"{runtime:.4f}")
                
                # Show tensors
                st.subheader("Tensor Elements 🔢")
                for i, node in enumerate(layer.layer_nodes):
                    st.text(f"Node {i}:")
                    st.code(node.tensor.numpy())

                # Create heatmaps for tensor elements
                with left_col:
                    st.subheader("Tensor Heatmaps 🌡️")
                    
                    # Create rows of two figures each
                    for i in range(0, len(layer.layer_nodes), 2):
                        col1, col2 = st.columns(2)
                        
                        # First figure in row
                        with col1:
                            node = layer.layer_nodes[i]
                            tensor_data = node.tensor.numpy()
                            tensor_shape = tensor_data.shape
                            
                            if len(tensor_shape) > 2:
                                tensor_data = tensor_data.reshape(-1, tensor_shape[-1])
                            
                            plt.style.use('dark_background')
                            fig = plt.figure(figsize=(8, 6))
                            plt.imshow(tensor_data, cmap='magma', aspect='auto')
                            plt.colorbar()
                            plt.title(f'Node {i} Tensor Elements', color='white')
                            plt.xlabel('Column Index', color='white')
                            plt.ylabel('Row Index', color='white')
                            st.pyplot(fig)
                            plt.close()
                        
                        # Second figure in row (if available)
                        with col2:
                            if i + 1 < len(layer.layer_nodes):
                                node = layer.layer_nodes[i + 1]
                                tensor_data = node.tensor.numpy()
                                tensor_shape = tensor_data.shape
                                
                                if len(tensor_shape) > 2:
                                    tensor_data = tensor_data.reshape(-1, tensor_shape[-1])
                                
                                plt.style.use('dark_background')
                                fig = plt.figure(figsize=(8, 6))
                                plt.imshow(tensor_data, cmap='magma', aspect='auto')
                                plt.colorbar()
                                plt.title(f'Node {i+1} Tensor Elements', color='white')
                                plt.xlabel('Column Index', color='white')
                                plt.ylabel('Row Index', color='white')
                                st.pyplot(fig)
                                plt.close()
            except Exception as e:
                st.error(f"An error occurred ❌: {str(e)}")