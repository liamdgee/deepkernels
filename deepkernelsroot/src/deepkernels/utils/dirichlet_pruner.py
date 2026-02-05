#--Dependencies---#
import os
import logging
import torch
import torch.nn as nn

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DirichletPruner:
    def __init__(self, model, **kwargs):
        self.model = model
        self.threshold = kwargs.get('dead_atom_weight_threshold', 5e-3)

    @torch.no_grad()
    def prune(self, local_idx):
        """Removes atoms that have collapsed from KL divergence penalisation term in optimisation & training"""
        weights = self.model.dirichlet.get_weights(local_idx)
        indices_live = torch.where(weights > self.threshold)[0]
        n_live_atoms = len(indices_live)
        n_dead_atoms = self.model.dirichlet.k_atoms - n_live_atoms

        if not n_dead_atoms:
            logger.info("Audit Status: all atoms are active -- No pruning is required")
            return
        
        else:
            logger.info(f"Surgery Required -- Pruning {n_dead_atoms} dead atoms from dirichlet 'menu'")
        
        #---Contunue if number of dead atoms is non-zero--#

        #--Slice params to only contain live data---#
        self._update_param('mu_atom', indices_live)
        self._update_param('log_sigma_atom', indices_live)

        #--Update global params for stick-breaking process in dirichlet module---#
        if hasattr(self.model.dirichlet, 'v_k'):
            v_len_sliced = max(1, n_live_atoms - 1)
            self.model.dirichlet.v_k = nn.Parameter(self.model.dirichlet.v_k[:v_len_sliced].clone())
        
        self.model.dirichlet.k_atoms = n_live_atoms
        
        logger.info(f"Surgery Complete --Pruned {n_dead_atoms} dead atoms. Active CRP menu: {n_live_atoms} atoms.")
    
    def _update_param(self, name, indices):
        p_old = getattr(self.model.dirichlet, name)
        p_data_update = p_old.data[indices].clone()
        setattr(self.model.dirichlet, name, nn.Parameter(p_data_update))

        
