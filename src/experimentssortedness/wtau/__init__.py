from itertools import repeat

from numpy import lexsort, array, empty
import pathos.multiprocessing as mp

from .wtau import _weightedrankedtau


# from .wtau import parwtau
def parwtau(scoresX, scoresX_, R, parallel=True):
    n = R.shape[1]
    m = len(scoresX)
    # ret = empty(n, dtype=float)
    lex = lexsort((scoresX_, scoresX))
    perms = (lex.copy() for i in range(n))
    exchanges_weights = (array([0.0]) for i in range(n))
    temps = (empty(m, dtype=int) for i in range(n))

    jobs = (
        repeat(scoresX), repeat(scoresX_), (R[:, i] for i in range(n)),
        repeat(m), perms, exchanges_weights, temps
    )
    tmap = mp.ProcessingPool().imap if parallel else map
    return array(list(tmap(_weightedrankedtau, *jobs)))

    # for i, r in enumerate(tmap(_weightedrankedtau, *jobs)):
    # ret[i] = _weightedrankedtau(n, exchanges_weight, perm, temp, scoresX, scoresX_, R[:, i])
    # ret[i] = r
