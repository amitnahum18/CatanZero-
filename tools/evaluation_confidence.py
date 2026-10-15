"""Exact fixed-sample acceptance criteria; not valid for optional stopping."""
from scipy.stats import beta


def win_rate_acceptance(wins, games, *, target=0.55, confidence=0.95):
    if games < 1 or not 0 <= wins <= games or not 0 < confidence < 1 or not 0 < target < 1:
        raise ValueError('valid nonempty binomial sample and confidence/target required')
    alpha = 1-confidence
    lower = 0. if wins == 0 else float(beta.ppf(alpha/2, wins, games-wins+1))
    upper = 1. if wins == games else float(beta.ppf(1-alpha/2, wins+1, games-wins))
    return {'wins':wins, 'games':games, 'win_rate':wins/games,
            'method':'two-sided exact Clopper-Pearson, predeclared fixed sample',
            'confidence':confidence, 'interval':[lower,upper], 'target':target,
            'passed':lower>target,
            'scope':'the tested fixed policy, simulator and opponent configuration only'}
