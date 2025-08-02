import tensorkrowch as tk
import torch
import numpy as np

class TN_general:
    '''Base class for tensor network layers.'''
    def __init__(self,
                n_nodes:int,
                name:str='Layer_Auto',
                max_tol:float=1e20,
                min_tol:float=1e-20,
                norm_factor:float=1):
        """Initialize tensor network layer.
        
        Args:
            num_nodes: Number of nodes in the network
            name: Name of the layer
            max_tolerance: Maximum tolerance for normalization
            min_tolerance: Minimum tolerance for normalization 
            normalization_factor: Normalization factor
        """
        self.n_nodes = n_nodes
        self.norm_factor = norm_factor
        self.max_tol = max_tol
        self.min_tol = min_tol

    def normalize(self,
                max_iterations:int=100,
                method:str='Frobenius'):
        """Normalize the tensor network initialization.
        
        Args:
            max_iterations: Maximum number of iterations allowed
            method: Normalization method ('Frobenius' or 'Linear')
            verbose: Whether to track number of steps
        """
        # Calculate initial norm
        if method == 'Frobenius':
            norm = self.compute_frob()
            exponent = 1/(2*self.n_nodes)
        elif method == 'Lineal':
            norm = self.compute_lineal() 
            exponent = 1/self.n_nodes

        norm /= self.norm_factor

        # Classify initial norm
        if np.isnan(norm):
            norm_state = 1  # nan
        elif self.min_tol < norm < self.max_tol:
            norm_state = 0  # in range
        elif norm == np.inf:
            norm_state = 1  # inf
        elif norm > self.max_tol:
            norm_state = 2  # greater
        elif norm == 0:
            norm_state = -1  # zero
        else:
            norm_state = -2  # lower

        # Initial normalization if finite
        if norm_state in (2, -2, 0):
            for node_idx in range(self.n_nodes):
                self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/(norm**exponent)
            norm_state = 0

        # Iterative normalization if needed
        if norm_state != 0:
            normalize_node0 = True
            self.steps = 1
        else:
            normalize_node0 = False
            self.steps = 0

        # Node 0 normalization
        while normalize_node0 and self.steps < max_iterations:
            partial_norm = self.compute_part_frob(0)
            partial_norm /= self.norm_factor

            if norm_state == 1 and partial_norm > self.max_tol:  # Infinite case
                if partial_norm == np.inf or np.isnan(partial_norm):
                    scaling_factor = (10*(1+np.random.rand()))**exponent
                    for node_idx in range(self.n_nodes):
                        self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/scaling_factor
                    self.steps += 1
                else:  # Finite case
                    scaling_factor = partial_norm**exponent
                    for node_idx in range(self.n_nodes):
                        self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/scaling_factor
                    partial_norm /= scaling_factor
                    normalize_node0 = False

            elif norm_state == -1 and partial_norm < self.min_tol:  # Zero case
                if partial_norm == 0:
                    scaling_factor = (0.1/(1+np.random.rand()))**exponent
                    for node_idx in range(self.n_nodes):
                        self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/scaling_factor
                    self.steps += 1
                else:  # Finite case
                    scaling_factor = partial_norm**exponent
                    for node_idx in range(self.n_nodes):
                        self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/scaling_factor
                    partial_norm /= scaling_factor
                    normalize_node0 = False
            else:  # In range
                normalize_node0 = False

        # Check norm after node 0
        norm, norm_state = self.classificate_norm(method)

        if norm_state in (2, -2, 0):
            scaling_factor = norm**exponent
            for node_idx in range(self.n_nodes):
                self.layer_nodes[node_idx].tensor = self.layer_nodes[node_idx].tensor/scaling_factor
            norm_state = 0

        # Normalize remaining nodes
        start_node = 1
        continue_normalizing = True
        while norm_state != 0 and self.steps <= max_iterations:
            normalize_final = True
            for node_idx in range(start_node, self.n_nodes-1):
                prev_norm = partial_norm
                if method == 'Frobenius':
                    partial_norm = self.compute_part_frob(node_idx)
                elif method == 'Lineal':
                    partial_norm = self.compute_part_lineal(node_idx)

                partial_norm /= self.norm_factor

                if norm_state == 1 and partial_norm > self.max_tol:  # Infinity case
                    normalize_final = False
                    if partial_norm == np.inf or np.isnan(partial_norm):
                        start_node = node_idx
                        partial_norm = prev_norm
                        scaling_factor = partial_norm**exponent
                        for j in range(self.n_nodes):
                            self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor
                        partial_norm /= scaling_factor**(node_idx)
                        self.steps += 1
                        continue_normalizing = False
                        break
                    else:
                        scaling_factor = partial_norm**exponent
                        for j in range(self.n_nodes):
                            self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor

                elif norm_state == -1 and partial_norm < self.min_tol:  # Zero case
                    normalize_final = False
                    if partial_norm == 0:
                        start_node = node_idx
                        partial_norm = prev_norm
                        scaling_factor = partial_norm**exponent
                        for j in range(self.n_nodes):
                            self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor
                        partial_norm /= scaling_factor**(node_idx)
                        self.steps += 1
                        continue_normalizing = False
                        break
                    else:
                        scaling_factor = partial_norm**exponent
                        for j in range(self.n_nodes):
                            self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor
                else:
                    pass

            # Final node normalization
            if normalize_final:
                final_node = self.n_nodes-2
                scaling_factor = partial_norm**exponent
                for j in range(self.n_nodes):
                    self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor
                partial_norm /= scaling_factor**(final_node+1)

            # Check final norm
            norm, norm_state = self.classificate_norm(method)

            if norm_state in (2, -2, 0):
                scaling_factor = norm**exponent
                for j in range(self.n_nodes):
                    self.layer_nodes[j].tensor = self.layer_nodes[j].tensor/scaling_factor
                norm, norm_state = self.classificate_norm(method)
                if norm_state != 0 and continue_normalizing:
                    self.steps += 1
            elif continue_normalizing:
                self.steps += 1

    def classificate_norm(self,
                        method:str='Frobenius'):
        """Classify the norm of the tensor network.
        
        Args:
            method: Normalization method
            
        Returns:
            norm: Calculated norm value
            norm_state: Classification of the norm (-2: lower, -1: zero, 0: in range, 
                   1: inf/nan, 2: greater)
        """
        if method == 'Frobenius':
            norm = self.compute_frob()
        elif method == 'Lineal':
            norm = self.compute_lineal()
                
        norm /= self.norm_factor

        if np.isnan(norm):
            norm_state = 1  # nan
        elif self.min_tol < norm < self.max_tol:
            norm_state = 0  # in range
        elif norm == np.inf:
            norm_state = 1  # inf
        elif norm > self.max_tol:
            norm_state = 2  # greater
        elif norm == 0:
            norm_state = -1  # zero
        else:
            norm_state = -2  # lower

        return norm, norm_state


class MPO(TN_general):
    """Matrix Product Operator (MPO) class that inherits from TN_general.
    
    An MPO represents a quantum operator as a tensor network of rank-4 tensors connected in a 1D chain.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes/tensors in the MPO chain
    phys_dim : int 
        Physical dimension of input indices
    bond_dim : int or list or numpy.ndarray
        Bond dimension(s) between adjacent tensors. If int, uses same dimension for all bonds.
        If list/array, specifies dimension for each bond separately.
    phys_out : int, optional
        Physical dimension of output indices. If 0, uses phys_dim. Default is 0.
    std : float, optional
        Standard deviation for random tensor initialization. Default is 0.5.
    mean : float, optional
        Mean for random tensor initialization. Default is 1.
    name : str, optional
        Name identifier for the MPO. Default is 'Layer_MPO'.
    max_tol : float, optional
        Maximum tolerance for normalization. Default is 1e20.
    min_tol : float, optional
        Minimum tolerance for normalization. Default is 1e-20.
    norm_factor : float, optional
        Normalization factor. Default is 1.
    method : str, optional
        Normalization method to use - either 'Frobenius' or 'Lineal'. Default is 'Frobenius'.
    """
    def __init__(self,
                n_nodes: int,
                phys_dim: int,
                bond_dim: int|list|np.ndarray,
                phys_out: int = 0,
                std: float = 0.5,
                mean: float = 1,
                name: str = 'Layer_MPO',
                max_tol: float = 1e20,
                min_tol: float = 1e-20,
                norm_factor: float = 1,
                method: str = 'Frobenius'):

        super().__init__(n_nodes=n_nodes, name=name, max_tol=max_tol, min_tol=min_tol, norm_factor=norm_factor)

        self.phys_dim = phys_dim
        self.phys_out = phys_out if phys_out != 0 else phys_dim
        self.bond_dim = [bond_dim] * (n_nodes-1) if isinstance(bond_dim, int) else list(bond_dim)
        self.std = std
        self.mean = mean
        self.name = name

        # Define shapes for each tensor in the MPO chain
        shape_list = [(self.phys_dim, self.bond_dim[0], self.phys_out)]
        for i in range(1, self.n_nodes-1):
            shape_list.append((self.phys_dim, self.bond_dim[i-1], self.bond_dim[i], self.phys_out))
        shape_list.append((self.phys_dim, self.bond_dim[-1], self.phys_out))

        # Initialize tensors based on normalization method
        if method == 'Frobenius':
            tensor_init = lambda shape: torch.normal(self.mean, self.std, size=shape, dtype=torch.float32)
        else:  # Lineal method
            tensor_init = lambda shape: torch.abs(torch.normal(self.mean, self.std, size=shape, dtype=torch.float32))

        # Create MPO nodes
        self.layer_nodes = [
            tk.Node(tensor=tensor_init(shape_list[0]), name='P_(0)', axes_names=['in','bond2','out'])
        ]
        
        for i in range(1, self.n_nodes-1):
            self.layer_nodes.append(
                tk.Node(tensor=tensor_init(shape_list[i]), 
                       name=f'P_({i})', 
                       axes_names=['in','bond1','bond2','out'])
            )
            
        self.layer_nodes.append(
            tk.Node(tensor=tensor_init(shape_list[-1]), 
                   name=f'P_({self.n_nodes-1})', 
                   axes_names=['in','bond1','out'])
        )

        # Normalize the MPO
        self.normalize(method=method)

    def compute_frob(self) -> float:
        """Calculate the Frobenius norm of the MPO by contracting with its conjugate."""
        tensors = [elem.copy() for elem in self.layer_nodes]
        tensors_copy = [elem.copy() for elem in self.layer_nodes]
            
        # Connect physical and bond indices
        for i in range(self.n_nodes):
            tensors[i]['in'] ^ tensors_copy[i]['in']
            tensors[i]['out'] ^ tensors_copy[i]['out']
            if i < self.n_nodes-1:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1'] 
                tensors_copy[i]['bond2'] ^ tensors_copy[i+1]['bond1']
            
        # Contract the network efficiently from left to right
        tensor = tk.contract_between(tensors[0], tensors_copy[0])
        for i in range(1, self.n_nodes):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, tensors_copy[i])

        return float(tensor.tensor)

    def compute_lineal(self) -> float:
        """Calculate the linear norm of the MPO by contracting with unit tensors."""
        tensors = [elem.copy() for elem in self.layer_nodes]
        unit_tensors_in = [tk.Node(tensor=torch.ones(self.phys_dim), 
                                name=f'1(1)_({i})', 
                                axes_names=['in']) for i in range(self.n_nodes)]
        unit_tensors_out = [tk.Node(tensor=torch.ones(self.phys_out),
                                 name=f'1(2)_({i})',
                                 axes_names=['in']) for i in range(self.n_nodes)]
        
        # Connect physical indices and bond indices
        for i in range(self.n_nodes):
            tensors[i]['in'] ^ unit_tensors_in[i]['in']
            tensors[i]['out'] ^ unit_tensors_out[i]['in']
            if i < self.n_nodes-1:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1']
            
        # Contract efficiently from left to right
        tensor = tk.contract_between(tensors[0], unit_tensors_in[0])
        tensor = tk.contract_between(tensor, unit_tensors_out[0])
        for i in range(1, self.n_nodes):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, unit_tensors_in[i])
            tensor = tk.contract_between(tensor, unit_tensors_out[i])

        return float(tensor.tensor)

    def compute_part_frob(self, n_nodes: int) -> float:
        """Calculate partial Frobenius norm up to given number of nodes."""
        if n_nodes == self.n_nodes:
            return self.compute_frob()

        tensors = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]
        tensors_copy = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]

        # Connect indices
        for i in range(n_nodes+1):
            tensors[i]['in'] ^ tensors_copy[i]['in']
            tensors[i]['out'] ^ tensors_copy[i]['out']
            if i < n_nodes:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1'] 
                tensors_copy[i]['bond2'] ^ tensors_copy[i+1]['bond1']
        
        tensors[-1]['bond2'] ^ tensors_copy[-1]['bond2']
                
        # Contract efficiently
        tensor = tk.contract_between(tensors[0], tensors_copy[0])
        for i in range(1, n_nodes+1):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, tensors_copy[i])

        return float(tensor.tensor)

    def compute_part_lineal(self, n_nodes: int) -> float:
        """Calculate partial linear norm up to given number of nodes."""
        if n_nodes == self.n_nodes:
            return self.compute_lineal()

        tensors = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]
        unit_tensors_in = [tk.Node(tensor=torch.ones(self.phys_dim),
                                name=f'1(1)_({i})',
                                axes_names=['in']) for i in range(n_nodes+1)]
        unit_tensors_out = [tk.Node(tensor=torch.ones(self.phys_out),
                                 name=f'1(2)_({i})',
                                 axes_names=['in']) for i in range(n_nodes+1)]
        unit_tensor_final = tk.Node(tensor=torch.ones(self.bond_dim[n_nodes]),
                              name=f'1(f)',
                              axes_names=['in'])

        # Connect indices
        for i in range(n_nodes+1):
            tensors[i]['in'] ^ unit_tensors_in[i]['in']
            tensors[i]['out'] ^ unit_tensors_out[i]['in']
            if i < n_nodes:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1']
        
        tensors[-1]['bond2'] ^ unit_tensor_final['in']
            
        # Contract efficiently
        tensor = tk.contract_between(tensors[0], unit_tensors_in[0])
        tensor = tk.contract_between(tensor, unit_tensors_out[0])
        for i in range(1, n_nodes+1):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, unit_tensors_in[i])
            tensor = tk.contract_between(tensor, unit_tensors_out[i])
        
        
        tensor = tk.contract_between(tensor, unit_tensor_final)
 
        return float(tensor.tensor)


class MPS(TN_general):
    """Matrix Product State (MPS) class that inherits from TN_general.
    
    An MPS represents a quantum state as a tensor network of rank-3 tensors connected in a 1D chain.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes/tensors in the MPS chain
    phys_dim : int 
        Physical dimension of each node (dimension of local Hilbert space)
    bond_dim : int or list or numpy.ndarray
        Bond dimension(s) between adjacent tensors. If int, uses same dimension for all bonds.
        If list/array, specifies dimension for each bond separately.
    std : float, optional
        Standard deviation for random tensor initialization. Default is 0.5.
    mean : float, optional
        Mean for random tensor initialization. Default is 1.
    name : str, optional
        Name identifier for the MPS. Default is 'Layer_MPS'.
    max_tol : float, optional
        Maximum tolerance for normalization. Default is 1e20.
    min_tol : float, optional
        Minimum tolerance for normalization. Default is 1e-20.
    norm_factor : float, optional
        Normalization factor. Default is 1.
    method : str, optional
        Normalization method to use - either 'Frobenius' or 'Lineal'. Default is 'Frobenius'.
    """
    def __init__(self, 
                 n_nodes: int,
                 phys_dim: int, 
                 bond_dim: int, 
                 std: float = 0.5,
                 mean: float = 1,
                 name: str = 'Layer_MPS',
                 max_tol: float = 1e20,
                 min_tol: float = 1e-20,
                 norm_factor: float = 1,
                 method: str = 'Frobenius'):

        super().__init__(n_nodes=n_nodes, name=name, max_tol=max_tol, min_tol=min_tol, norm_factor=norm_factor)

        self.phys_dim = phys_dim
        self.bond_dim = [bond_dim] * (n_nodes-1) if isinstance(bond_dim, int) else list(bond_dim)
        self.std = std
        self.mean = mean
        self.name = name

        # Define shapes for each tensor in the MPS chain
        shape_list = [(self.phys_dim, self.bond_dim[0])]
        for i in range(1, self.n_nodes-1):
            shape_list.append((self.phys_dim, self.bond_dim[i-1], self.bond_dim[i]))
        shape_list.append((self.phys_dim, self.bond_dim[-1]))

        # Initialize tensors based on normalization method
        if method == 'Frobenius':
            tensor_init = lambda shape: torch.normal(self.mean, self.std, size=shape, dtype=torch.float32)
        else:  # Lineal method
            tensor_init = lambda shape: torch.abs(torch.normal(self.mean, self.std, size=shape, dtype=torch.float32))

        # Create MPS nodes
        self.layer_nodes = [
            tk.Node(tensor=tensor_init(shape_list[0]), name='P_(0)', axes_names=['in','bond2'])
        ]
        
        for i in range(1, self.n_nodes-1):
            self.layer_nodes.append(
                tk.Node(tensor=tensor_init(shape_list[i]), 
                       name=f'P_({i})', 
                       axes_names=['in','bond1','bond2'])
            )
            
        self.layer_nodes.append(
            tk.Node(tensor=tensor_init(shape_list[-1]), 
                   name=f'P_({self.n_nodes-1})', 
                   axes_names=['in','bond1'])
        )

        # Normalize the MPS
        self.normalize(method=method)

    def compute_frob(self) -> float:
        """Calculate the Frobenius norm of the MPS by contracting with its conjugate."""
        tensors = [elem.copy() for elem in self.layer_nodes]
        tensors_copy = [elem.copy() for elem in self.layer_nodes]
            
        # Connect physical and bond indices
        for i in range(self.n_nodes):
            tensors[i]['in'] ^ tensors_copy[i]['in']
            if i < self.n_nodes-1:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1'] 
                tensors_copy[i]['bond2'] ^ tensors_copy[i+1]['bond1']
            
        # Contract the network efficiently from left to right
        tensor = tk.contract_between(tensors[0], tensors_copy[0])
        for i in range(1, self.n_nodes):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, tensors_copy[i])

        return float(tensor.tensor)

    def compute_lineal(self) -> float:
        """Calculate the linear norm of the MPS by contracting with unit tensors."""
        tensors = [elem.copy() for elem in self.layer_nodes]
        unit_tensors = [tk.Node(tensor=torch.ones(self.phys_dim), 
                              name=f'1(1)_({i})', 
                              axes_names=['in']) for i in range(self.n_nodes)]
        
        # Connect physical indices and bond indices
        for i in range(self.n_nodes):
            tensors[i]['in'] ^ unit_tensors[i]['in']
            if i < self.n_nodes-1:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1']
            
        # Contract efficiently from left to right
        tensor = tk.contract_between(tensors[0], unit_tensors[0])
        for i in range(1, self.n_nodes):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, unit_tensors[i])

        return float(tensor.tensor)

    def compute_part_frob(self, n_nodes: int) -> float:
        """Calculate partial Frobenius norm up to given number of nodes."""
        if n_nodes == self.n_nodes:
            return self.compute_frob()

        tensors = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]
        tensors_copy = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]

        # Connect indices
        for i in range(n_nodes+1):
            tensors[i]['in'] ^ tensors_copy[i]['in']
            if i < n_nodes:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1'] 
                tensors_copy[i]['bond2'] ^ tensors_copy[i+1]['bond1']
        
        tensors[-1]['bond2'] ^ tensors_copy[-1]['bond2']
                
        # Contract efficiently
        tensor = tk.contract_between(tensors[0], tensors_copy[0])
        for i in range(1, n_nodes+1):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, tensors_copy[i])

        return float(tensor.tensor)

    def compute_part_lineal(self, n_nodes: int) -> float:
        """Calculate partial linear norm up to given number of nodes."""
        if n_nodes == self.n_nodes:
            return self.compute_lineal()

        tensors = [elem.copy() for elem in self.layer_nodes[0:n_nodes+1]]
        unit_tensors = [tk.Node(tensor=torch.ones(self.phys_dim),
                              name=f'1(1)_({i})',
                              axes_names=['in']) for i in range(n_nodes+1)]
        unit_tensor_final = tk.Node(tensor=torch.ones(self.bond_dim[n_nodes]),
                              name=f'1(f)',
                              axes_names=['in'])

        # Connect indices
        for i in range(n_nodes+1):
            tensors[i]['in'] ^ unit_tensors[i]['in']
            if i < n_nodes:
                tensors[i]['bond2'] ^ tensors[i+1]['bond1']
        
        tensors[-1]['bond2'] ^ unit_tensor_final['in']
            
        # Contract efficiently
        tensor = tk.contract_between(tensors[0], unit_tensors[0])
        for i in range(1, n_nodes+1):
            tensor = tk.contract_between(tensor, tensors[i])
            tensor = tk.contract_between(tensor, unit_tensors[i])

        tensor = tk.contract_between(tensor, unit_tensor_final)
 
        return float(tensor.tensor)
