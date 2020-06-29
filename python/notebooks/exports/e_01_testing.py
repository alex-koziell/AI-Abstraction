# module automatically generated from 01_testing.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

import operator
from functools import partial
import torch

def assert_comparable(a, b, cmp, cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b), f'{a} !{cname} {b}'
    
def test_equal(a, b): 
    try: 
        assert_comparable(a, b, operator.eq, '==')
        print(f'Argument `{a}` IS equal to `{b}`.')
    except:
        print(f'Argument `{a}` IS NOT equal to `{b}`.')
    
def test_near_torch(a, b):
    try:
        assert_comparable(a, b, 
                          partial(torch.allclose, rtol=1e-3, atol=1e-5),
                          "torch.allclose")
        print(f'Arguments ARE near.')
    except:
        print(f'Arguments ARE NOT near.')