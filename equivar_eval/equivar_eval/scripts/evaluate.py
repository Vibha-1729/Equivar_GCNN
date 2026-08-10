import sys
import logging
import importlib.metadata
import time
import math
import numpy
import torch
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
import csv
from equivar_eval.config import g_config
from equivar_eval.process import AtomsToGraphs,InMemoryDatasetUtil

def register_jit_custom_ops():
    csr_single_ops = [
        ("torch_scatter::segment_sum_csr", "(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor", "sum"),
        ("torch_scatter::segment_mean_csr", "(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor", "mean"),
    ]
    for op_name, schema, reduce_op in csr_single_ops:
        try:
            torch.library.define(op_name, schema)
        except Exception:
            pass
        def make_csr_single_impl(r_op):
            def csr_single_impl(src: torch.Tensor, indptr: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
                counts = indptr[1:] - indptr[:-1]
                index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
                res = scatter(src, index, dim=0, reduce=r_op)
                if out is not None:
                    out.copy_(res)
                    return out
                return res
            return csr_single_impl
        torch.library.impl(op_name, "default")(make_csr_single_impl(reduce_op))

    csr_tuple_ops = [
        ("torch_scatter::segment_min_csr", "(Tensor src, Tensor indptr, Tensor? out=None) -> (Tensor, Tensor)", "min"),
        ("torch_scatter::segment_max_csr", "(Tensor src, Tensor indptr, Tensor? out=None) -> (Tensor, Tensor)", "max"),
    ]
    for op_name, schema, reduce_op in csr_tuple_ops:
        try:
            torch.library.define(op_name, schema)
        except Exception:
            pass
        def make_csr_tuple_impl(r_op):
            def csr_tuple_impl(src: torch.Tensor, indptr: torch.Tensor, out: torch.Tensor = None):
                counts = indptr[1:] - indptr[:-1]
                index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
                res = scatter(src, index, dim=0, reduce=r_op)
                arg_idx = torch.zeros_like(res, dtype=torch.long)
                if out is not None:
                    out.copy_(res)
                    return out, arg_idx
                return res, arg_idx
            return csr_tuple_impl
        torch.library.impl(op_name, "default")(make_csr_tuple_impl(reduce_op))

    scatter_ops = [
        ("torch_scatter::scatter_sum", "(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> Tensor", "sum"),
        ("torch_scatter::scatter_mul", "(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> Tensor", "mul"),
        ("torch_scatter::scatter_mean", "(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> Tensor", "mean"),
        ("torch_scatter::scatter_min", "(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)", "min"),
        ("torch_scatter::scatter_max", "(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)", "max"),
    ]
    for op_name, schema, reduce_op in scatter_ops:
        try:
            torch.library.define(op_name, schema)
        except Exception:
            pass
        if "min" in op_name or "max" in op_name:
            def make_scatter_tuple_impl(r_op):
                def scatter_tuple_impl(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None, dim_size: int = None):
                    res = scatter(src, index, dim=dim, dim_size=dim_size, reduce=r_op)
                    arg_idx = torch.zeros_like(index)
                    if out is not None:
                        out.copy_(res)
                        return out, arg_idx
                    return res, arg_idx
                return scatter_tuple_impl
            torch.library.impl(op_name, "default")(make_scatter_tuple_impl(reduce_op))
        else:
            def make_scatter_impl(r_op):
                def scatter_impl(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None, dim_size: int = None):
                    res = scatter(src, index, dim=dim, dim_size=dim_size, reduce=r_op)
                    if out is not None:
                        out.copy_(res)
                        return out
                    return res
                return scatter_impl
            torch.library.impl(op_name, "default")(make_scatter_impl(reduce_op))

    for op_name, schema in [
        ("torch_scatter::gather_csr", "(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor"),
        ("torch_scatter::broadcast", "(Tensor src, Tensor ref, int dim) -> Tensor"),
        ("torch_sparse::ptr2ind", "(Tensor ptr, int max_size) -> Tensor"),
        ("torch_sparse::ind2ptr", "(Tensor ind, int M) -> Tensor"),
    ]:
        try:
            torch.library.define(op_name, schema)
        except Exception:
            pass

    @torch.library.impl("torch_scatter::gather_csr", "default")
    def gather_csr_impl(src: torch.Tensor, indptr: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        counts = indptr[1:] - indptr[:-1]
        index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
        res = src[index]
        if out is not None:
            out.copy_(res)
            return out
        return res

    @torch.library.impl("torch_scatter::broadcast", "default")
    def broadcast_impl(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
        return src

    @torch.library.impl("torch_sparse::ptr2ind", "default")
    def ptr2ind_impl(ptr: torch.Tensor, max_size: int) -> torch.Tensor:
        counts = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(torch.arange(len(counts), device=ptr.device), counts)

    @torch.library.impl("torch_sparse::ind2ptr", "default")
    def ind2ptr_impl(ind: torch.Tensor, M: int) -> torch.Tensor:
        return torch.ops.torch_geometric.ind2ptr(ind, M)

register_jit_custom_ops()

SQ2_1=1.0/math.sqrt(2.0)
SQ3_1=1.0/math.sqrt(3.0)
SQ23_1=SQ2_1*SQ3_1

def write_csv(datas,path_out):
    '''Write results to a csv file
    '''
    csvwriter=csv.writer(open(path_out,'w'))
    N=datas.shape[1]-1
    csvwriter.writerow(['ids']+['prediction']*N)
    [csvwriter.writerow(_data) for _data in datas]

def _ctime(secs=None):
    return time.asctime(time.localtime(secs))

def main():

    _tstart=time.time()
    logging.basicConfig(handlers=[logging.FileHandler('equivar_eval.log'),logging.StreamHandler(sys.stdout)],
            level=logging.DEBUG,format='%(levelname)s %(message)s')
    logging.info(f'##### starting module {__name__} at {_ctime(_tstart)} #####')
    _ver=importlib.metadata.version('equivar_eval')
    logging.info(f'equivar_eval version {_ver}')

    cuda_device_count=torch.cuda.device_count()
    if cuda_device_count==0:
        device='cpu'
        logging.info(f'Running on CPU (device={device})')
    elif cuda_device_count==1:
        device='cuda'
        logging.info(f'Running on a single GPU (device={device})')
    else:
        device='cuda'
        logging.warning(f'{cuda_device_count} GPUs are present, but parallel job is not implemented yet, running on a single GPU (device={device})')

    cob=torch.tensor(
    [
        [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,  -SQ2_1, 0.0,   0.0,   0.0,  ],     # t11
        [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,   SQ2_1,],     # t12
        [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,  -SQ2_1, 0.0,  ],     # t13
        [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,  -SQ2_1,],     # t21
        [SQ3_1, 0.0,   0.0,   2.0*SQ23_1, 0.0,   0.0,   0.0,   0.0,   0.0,  ],     # t22
        [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,   SQ2_1, 0.0,   0.0,  ],     # t23
        [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,   SQ2_1, 0.0,  ],     # t31
        [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,  -SQ2_1, 0.0,   0.0,  ],     # t32
        [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,   SQ2_1, 0.0,   0.0,   0.0,  ],     # t33
    ],dtype=torch.get_default_dtype()
    ).T
    cob=cob.to(device)

    _saved_model_path=g_config['saved_model_path']
    logging.info(f'loading model from \'{_saved_model_path}\'')
    model=torch.jit.load(_saved_model_path,map_location=torch.device(device))
    logging.info('Done')
    model.eval()
    logging.info(f'Converting input data')

    graph_max_radius=g_config['graph_max_radius'] if 'graph_max_radius' in g_config else 3.0
    num_radial=g_config['num_radial'] if 'num_radial' in g_config else 32
    edge_sh_lmax=g_config['edge_sh_lmax'] if 'edge_sh_lmax' in g_config else 2
    radial_basis=g_config['radial_basis'] if 'radial_basis' in g_config else None
    a2g=AtomsToGraphs(
            path_in=g_config['data_dir'],
            graph_max_radius=graph_max_radius,
            num_radial=num_radial,
            edge_sh_lmax=edge_sh_lmax,
            radial_basis=radial_basis,
            )
    data,slices=a2g.convert()
    dataset=InMemoryDatasetUtil(data,slices)

    logging.info('Done')
    _batch_size=g_config['batch_size'] if 'batch_size' in g_config else 10
    logging.info(f'dataset size: {len(dataset)} batch size: {_batch_size}')
    predictions=[]
    data_loader=DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=False,
            pin_memory=True
            )
    logging.info('evaluating...')
    with torch.no_grad():
        for i,data in enumerate(data_loader):
            data=data.to(device)
            data_dict=data.to_dict()
            if '_num_nodes' in data_dict:
                # torch geometric saves value at this key as a list, while jit expects a tensor
                data_dict['_num_nodes']=torch.tensor(data_dict['_num_nodes'],dtype=data_dict['_num_nodes'][0].dtype)
            out=model(data_dict)
            out=out@cob
            # remove any 'drift' from Born charges, i.e., enforce charge neutrality
            # acoustic sum rule, eq 46 in Gonze97, or eq 6.4 in Pick70
            _s=scatter(out,data_dict['batch'],dim=0,reduce='sum')/data_dict['Natoms'][:,None]
            _index=torch.unsqueeze(data_dict['batch'],1).expand(-1,out.shape[1])
            out=out-torch.gather(_s,dim=0,index=_index)
            _o=out.cpu().numpy()
            predictions=_o if i==0 else numpy.vstack((predictions,_o))
            ids_temp=[_id_atom for _id_atom in data.structure_id.cpu().numpy()]
            if i==0:
                ids=ids_temp
            else:
                ids+=ids_temp
    predictions=numpy.column_stack((ids,predictions))
    _ouput_path=g_config['ouput_path']
    logging.info(f'writing model outputs to \'{_ouput_path}\'')
    write_csv(predictions,_ouput_path)
    _tend=time.time()
    logging.info(f'***** Running wall time: {_tend-_tstart:.2f} s *****')
    logging.info(f'***** {__name__} finished at {_ctime(_tend)} *****')

if __name__=='__main__':
    main()
