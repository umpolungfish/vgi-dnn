# VGI/sources/grammar_training_v1/dialetheic_beam.py
import torch
import torch.nn.functional as F

class Cut:
    T = 0b0001  # Constructive True (Affirmative)
    F = 0b0010  # Constructive False (Refutation)
    t = 0b0100  # Informational acceptable (Support)
    f = 0b1000  # Informational rejectable (Absence)

def engage(state: int) -> int:
    """⊞ (ENGAGR): If constructive weight exists, activate the informational layer."""
    if state & (Cut.T | Cut.F):
        return state | Cut.t | Cut.f
    return state

def coreflect(state: int) -> int:
    """The Classical Coreflector: Drops the informational layer."""
    return state & (Cut.T | Cut.F)

def info_mass(state: int) -> float:
    """Calculates the informational mass (0.0 to 1.0)."""
    return bin(state & (Cut.t | Cut.f)).count('1') / 2.0

def state_name(state: int) -> str:
    names = []
    if state & Cut.T: names.append('T')
    if state & Cut.F: names.append('F')
    if state & Cut.t: names.append('t')
    if state & Cut.f: names.append('f')
    return f"{''.join(names) if names else 'N'}"

class DialetheicScorer:
    def __init__(self, tokenizer, temperature: float = 0.8, penalty_factor: float = 0.1, reward_factor: float = 2.5):
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.penalty_factor = penalty_factor
        self.reward_factor = reward_factor
        
        # Pre-compute token masks for structural verdicts
        self.collapse_tokens = set()
        self.paradox_tokens = set()
        self.neutral_tokens = set()
        
        self._build_token_masks()

    def _build_token_masks(self):
        """Scans the vocabulary to find tokens that imply classical collapse or paradox sustainment."""
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            lower = token_str.lower().strip()
            
            # Classical collapse indicators (Forces coreflection to T or F)
            if any(k in lower for k in ['true', 'false', 'yes', 'no', 'always', 'never', 'impossible', 'certain', 'exactly']):
                self.collapse_tokens.add(token_id)
                
            # Paradox sustainment indicators (Engages the B-state / Glue)
            elif any(k in lower for k in ['both', 'neither', 'paradox', 'contradiction', 'sustain', 'hold', 'glue', 'ambivalent', 'dialetheic', 'simultaneously']):
                self.paradox_tokens.add(token_id)
                
            else:
                self.neutral_tokens.add(token_id)

    def score(self, logits: torch.Tensor, current_state: int) -> torch.Tensor:
        """
        Replaces standard F.softmax. Scores tokens based on their ability to 
        maintain the paraconsistent ambient (SIXTEEN_3) without collapsing.
        """
        # Classical baseline
        probs = F.softmax(logits / self.temperature, dim=-1)
        scores = probs.clone()
        
        # THE DIALETHEIC MODIFIER:
        # If the current state has constructive weight (T or F or both), 
        # classical resolution is penalized because it represents information loss.
        if current_state & (Cut.T | Cut.F): 
            for i in range(len(scores)):
                if i in self.collapse_tokens:
                    # Penalize tokens that force a classical collapse (coreflection)
                    scores[i] *= self.penalty_factor
                elif i in self.paradox_tokens:
                    # Boost tokens that sustain the paradox (engage the B state)
                    scores[i] *= self.reward_factor
                    
        # Renormalize to maintain a valid probability distribution for sampling
        return scores / scores.sum()

    def update_state(self, current_state: int, token_id: int) -> int:
        """Determines the next SIXTEEN_3 state based on the selected token."""
        if token_id in self.paradox_tokens:
            # The token actively holds the contradiction. Engage the informational layer.
            return engage(current_state)
        elif token_id in self.collapse_tokens:
            # The token forces a classical resolution. Coreflect the state.
            return coreflect(current_state)
        else:
            # Neutral token. Maintain the current structural state.
            return current_state