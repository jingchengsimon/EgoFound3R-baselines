"""Remote CPU regression gate against the installed legacy PyTorch3D kernel."""
import importlib.util
import sys
import torch
from formal_evaluation.contact.bvh_compat import LegacyCompatibleBVH


def check(kernel):
    old = kernel._point_to_mesh_distances
    fast = LegacyCompatibleBVH(old, kernel)
    results = []
    generator = torch.Generator().manual_seed(729)
    for size in [0.000072, 0.00008, 0.00010, 0.00011, 0.001, 1.]:
        for offset in [0., 1.25]:
            vertices = torch.tensor([[0.,0.,0.],[size,0.,0.],[0.,size,0.]]) + offset
            mesh = kernel._valid_mesh((vertices, torch.tensor([[0,1,2]])), device=torch.device('cpu'))
            if mesh is None:
                continue
            points = torch.rand((128,3), generator=generator) * size + offset
            boundary = torch.tensor([[size/4,size/4,z] for z in
                [0.,1e-6,.0139999,.014,.0140001,.0179999,.018,.0180001]]) + offset
            points = torch.cat([points,boundary]); mask = torch.ones(len(points),dtype=torch.bool)
            expected, vm = old(points,mask,mesh); actual, va = fast(points,mask,mesh)
            assert torch.equal(vm,va)
            error = float((expected-actual).abs().max())
            assert error <= 1e-6, (size,offset,error)
            assert torch.equal(expected <= .014, actual <= .014), (size,offset,'14mm')
            assert torch.equal(expected <= .018, actual <= .018), (size,offset,'18mm')
            results.append(dict(size=size,offset=offset,points=len(points),max_abs_m=error))
            fast.clear()
    # Mixed regular and degenerate triangles must combine their minima.
    verts=torch.tensor([[0.,0.,0.],[.00008,0.,0.],[0.,.00008,0.],
                        [0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    mesh=kernel._valid_mesh((verts,torch.tensor([[0,1,2],[3,4,5]])),device=torch.device('cpu'))
    points=torch.tensor([[.00002,.00002,0.],[.25,.25,1.014],[.00002,.00002,.014]])
    mask=torch.ones(3,dtype=torch.bool)
    a,av=old(points,mask,mesh);b,bv=fast(points,mask,mesh)
    assert torch.equal(av,bv) and torch.allclose(a,b,atol=1e-6,rtol=0)
    assert torch.equal(a<=.014,b<=.014) and torch.equal(a<=.018,b<=.018)
    results.append(dict(mixed=True,points=3,max_abs_m=float((a-b).abs().max())))
    return dict(status='passed',cases=results,stats=fast.stats)
