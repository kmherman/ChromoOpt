"""
This class is used for defining the scoring function(s) which can be used during
fine-tuning.
"""
# load general packages and functions
from collections import namedtuple
import torch
from rdkit import DataStructs
from rdkit.Chem import QED, AllChem, MolToSmiles
import numpy as np
import sklearn
import pandas as pd
from ProcessData import one_hot_features, get_adj_matrix
from PhotochemModel import GCNChem


class ScoringFunction:
    """
    A class for defining the scoring function components.
    """
    def __init__(self, constants : namedtuple) -> None:
        """
        Args:
        ----
            constants (namedtuple) : Contains job parameters as well as global
                                     constants.
        """
        self.score_components = constants.score_components  # list
        self.score_type       = constants.score_type        # list
        self.qsar_models      = constants.qsar_models       # dict
        self.device           = constants.device
        self.max_n_nodes      = constants.max_n_nodes
        self.score_thresholds = constants.score_thresholds

        self.n_graphs         = None  # placeholder

        assert len(self.score_components) == len(self.score_thresholds), \
               "`score_components` and `score_thresholds` do not match."

    def compute_score(self, graphs : list, termination : torch.Tensor,
                      validity : torch.Tensor, uniqueness : torch.Tensor) -> \
                      torch.Tensor:
        """
        Computes the overall score for the input molecular graphs.

        Args:
        ----
            graphs (list)              : Contains molecular graphs to evaluate.
            termination (torch.Tensor) : Termination status of input molecular
                                         graphs.
            validity (torch.Tensor)    : Validity of input molecular graphs.
            uniqueness (torch.Tensor)  : Uniqueness of input molecular graphs.

        Returns:
        -------
            final_score (torch.Tensor) : The final scores for each input graph.
        """
        self.n_graphs          = len(graphs)
        contributions_to_score = self.get_contributions_to_score(graphs=graphs)

        if len(self.score_components) == 1:
            final_score = contributions_to_score[0]

        elif self.score_type == "continuous":
            final_score = contributions_to_score[0]
            for component in contributions_to_score[1:]:
                final_score *= component

        elif self.score_type == "binary":
            component_masks = []
            for idx, score_component in enumerate(contributions_to_score):
                component_mask = torch.where(
                    score_component > self.score_thresholds[idx],
                    torch.ones(self.n_graphs, device=self.device, dtype=torch.uint8),
                    torch.zeros(self.n_graphs, device=self.device, dtype=torch.uint8)
                )
                component_masks.append(component_mask)

            final_score = component_masks[0]
            for mask in component_masks[1:]:
                final_score *= mask
                final_score  = final_score.float()

        else:
            raise NotImplementedError

        # remove contribution of duplicate molecules to the score
        final_score *= uniqueness

        # remove contribution of invalid molecules to the score
        final_score *= validity

        # remove contribution of improperly-terminated molecules to the score
        final_score *= termination

        return final_score

    def get_contributions_to_score(self, graphs : list) -> list:
        """
        Returns the different elements of the score.

        Args:
        ----
            graphs (list) : Contains molecular graphs to evaluate.

        Returns:
        -------
            contributions_to_score (list) : Contains elements of the score due to
                                            each scoring function component.
        """
        contributions_to_score = []

        for score_component in self.score_components:
            if "target_size" in score_component:

                target_size  = int(score_component[12:])

                assert target_size <= self.max_n_nodes, \
                       "Target size > largest possible size (`max_n_nodes`)."
                assert 0 < target_size, "Target size must be greater than 0."

                target_size *= torch.ones(self.n_graphs, device=self.device)
                n_nodes      = torch.tensor([graph.n_nodes for graph in graphs],
                                            device=self.device)
                max_nodes    = self.max_n_nodes
                score        = (
                    torch.ones(self.n_graphs, device=self.device)
                    - torch.abs(n_nodes - target_size)
                    / (max_nodes - target_size)
                )

                contributions_to_score.append(score)

            elif score_component == "QED":
                mols = [graph.molecule for graph in graphs]

                # compute the QED score for each molecule (if possible)
                qed = []
                for mol in mols:
                    try:
                        qed.append(QED.qed(mol))
                    except:
                        qed.append(0.0)
                score = torch.tensor(qed, device=self.device)

                contributions_to_score.append(score)

            elif "activity" in score_component:
                mols = [graph.molecule for graph in graphs]

                # `score_component` has to be the key to the QSAR model in the
                # `self.qsar_models` dict
                #qsar_model = self.qsar_models[score_component]
                score      = 0#self.compute_activity(mols, qsar_model)
		#score = 0
                contributions_to_score.append(score)

            elif "s1" in score_component:
                mols = [graph.molecule for graph in graphs]
                chromo_list = []
                solv_list = []
                for mol in mols:
                    try:
                        chromo_smi = MolToSmiles(mol)
                        if chromo_smi == ' ':
                            pass
                        else:
                            solv_smi = 'Cc1ccccc1'  #default: Toluene
                            chromo_list.append(str(chromo_smi))
                            solv_list.append(str(solv_smi))
                    except:
                        pass
                solv_sq_f = one_hot_features(solv_list)
                solv_sq_a = get_adj_matrix(solv_list)
                sq_f = one_hot_features(chromo_list)
                sq_a = get_adj_matrix(chromo_list)
                model = GCNChem().eval()
                torch.cuda.empty_cache()
                model.load_state_dict(torch.load('graphinvent/test_noheavyatoms.pt'))
                #model = torch.load('graphinvent/best_model_noheavyatoms.pt')
                model.cuda()
                model.eval()
                with torch.no_grad():
                    output = model(sq_f, sq_a, solv_sq_f, solv_sq_a)
                output[:, 0] *= 82.09614
                output[:, 0] += 401.65305
                print(output[:,0])
                output[:, 0] /= 100
                output[:, 1] *= 89.79511
                output[:, 1] += 499.41762
                output[:, 1] /= 100

                score = torch.tensor(output[:, 0], device=self.device)

                contributions_to_score.append(score)
            
            else:
                raise NotImplementedError("The score component is not defined. "
                                          "You can define it in "
                                          "`ScoringFunction.py`.")

        return contributions_to_score
