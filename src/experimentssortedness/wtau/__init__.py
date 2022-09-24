# from itertools import repeat
#
# from numpy import lexsort, array, empty
# import pathos.multiprocessing as mp
#
# from .wtau import _weightedrankedtau0
#
# import os
#
# from ..parallel import parallel_map
#
#
# # from .wtau import parwtau
# def parwtau(scoresX, scoresX_, R, parallel=True):
#     n = R.shape[1]
#     m = len(scoresX)
#     lex = lexsort((scoresX_, scoresX))
#
#     # perms = (lex.copy() for i in range(n))
#     # exchanges_weights = (array([0.0]) for i in range(n))
#     # temps = (empty(m, dtype=int) for i in range(n))
#     # jobs = (
#     #     repeat(scoresX), repeat(scoresX_), (R[:, i] for i in range(n)),
#     #     repeat(m), perms, exchanges_weights, temps
#     # )
#     # tmap = mp.ProcessingPool().imap if parallel else map
#     # r = list(tmap(_weightedrankedtau, *jobs))
#
#     jobs = ((R[i], lex.copy()) for i in range(n))
#     r = parallel_map(
#         lambda Ri, lexCopy: _weightedrankedtau0(scoresX, scoresX_, Ri, m, lexCopy, array([0.0]), empty(m, dtype=int)),
#         jobs,
#         threads=200
#     )
#
#     return array(r)
#
#     # for i, r in enumerate(tmap(_weightedrankedtau, *jobs)):
#     # ret[i] = _weightedrankedtau(n, exchanges_weight, perm, temp, scoresX, scoresX_, R[:, i])
#     # ret[i] = r
