import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf

def get_idx(i,j,dim, dim2=None):
    if dim2 is None:
        dim2 = dim
    return dim2*i+j+1

def get_neighbors(i,j,dim, dim2=None):
    if dim2 is None:
        dim2 = dim
    ret = []
    d = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for x,y in d:
        ii = i+x
        jj = j+y
        if ii >= 0 and jj >= 0 and ii < dim and jj < dim2:
            ret.append((ii,jj))
    return ret

def conjunction(vars):
    clause = None
    for v in vars:
        if clause is None:
            clause = v
        else:
            clause = clause & v
    return clause

# assumes output is dim*dim square
def build_constraints(dim, save_path=None):
    sdd = SddManager(var_count=dim**2, auto_gc_and_minimize=True)
    # alpha = sdd.vars[get_idx(0,0,dim)]
    # alpha = sdd.vars[get_idx(dim-1,dim-1,dim)]
    alpha = None
    # alpha.ref()
    for i in range(1,dim):
        for j in range(1,dim):
            print(i,j)
            if (i==dim-1 and j==dim-1):
                # continue
            # elif (i==0 and j==0):
                nbrs = get_neighbors(i,j,dim)
                clause = None
                for a in range(len(nbrs)):
                    off_clause = conjunction([-sdd.vars[get_idx(nbrs[k][0],nbrs[k][1],dim)] for k in range(len(nbrs)) if k != a])
                    off_clause.ref()
                    cur_clause = sdd.vars[get_idx(nbrs[a][0],nbrs[a][1],dim)] & off_clause
                    cur_clause.ref()
                    if clause is None:
                        clause = cur_clause
                        clause.ref()
                    else:
                        prev_clause = clause
                        clause = clause | cur_clause
                        clause.ref()
                        prev_clause.deref()
                    cur_clause.deref()
                    off_clause.deref()
                prev_alpha = alpha
                alpha = alpha & clause
                alpha.ref()
                prev_alpha.deref()
            else:
                nbrs = get_neighbors(i,j,dim)
                clause = None
                for a in range(len(nbrs)-1):
                    for b in range(a+1,len(nbrs)):
                        off_clause = conjunction([-sdd.vars[get_idx(nbrs[k][0],nbrs[k][1],dim)] for k in range(len(nbrs)) if (k != a and k != b)])
                        off_clause.ref()
                        cur_clause = sdd.vars[get_idx(nbrs[a][0],nbrs[a][1],dim)] & sdd.vars[get_idx(nbrs[b][0],nbrs[b][1],dim)] \
                            & off_clause
                        cur_clause.ref()
                        if clause is None:
                            clause = cur_clause
                            clause.ref()
                        else:
                            prev_clause = clause
                            clause = clause | cur_clause
                            clause.ref()
                            prev_clause.deref()
                        cur_clause.deref()
                        off_clause.deref()
                cur_alpha = -sdd.vars[get_idx(i,j,dim)] | clause
                if alpha is None:
                    alpha = cur_alpha
                    alpha.ref()
                else:
                    prev_alpha = alpha
                    alpha = alpha & cur_alpha
                    alpha.ref()
                    prev_alpha.deref()
    print(f"Model Count: {alpha.wmc(log_mode=False).propagate()}")
    if save_path is not None:
        alpha.save(str.encode(f'{save_path}.sdd'))
        alpha.vtree().save(str.encode(f'{save_path}.vtree'))
    return alpha
    
def build_inner_constraint(save_path):
    sdd = SddManager(var_count=9, auto_gc_and_minimize=True)
    nbrs = get_neighbors(1,1,3)
    clause = None
    for a in range(len(nbrs)-1):
        for b in range(a+1,len(nbrs)):
            off_clause = conjunction([-sdd.vars[get_idx(nbrs[k][0],nbrs[k][1],3)] for k in range(len(nbrs)) if (k != a and k != b)])
            off_clause.ref()
            cur_clause = sdd.vars[get_idx(nbrs[a][0],nbrs[a][1],3)] & sdd.vars[get_idx(nbrs[b][0],nbrs[b][1],3)] \
                & off_clause
            cur_clause.ref()
            if clause is None:
                clause = cur_clause
                clause.ref()
                cur_clause.deref()
            else:
                prev_clause = clause
                clause = clause | cur_clause
                clause.ref()
                prev_clause.deref()
                cur_clause.deref()
            off_clause.deref()
    alpha = -sdd.vars[get_idx(1,1,3)] | clause
    alpha.ref()
    print(f"Model Count: {alpha.wmc(log_mode=False).propagate()}")
    if save_path is not None:
        alpha.save(str.encode(f'{save_path}/inner.sdd'))
        alpha.vtree().save(str.encode(f'{save_path}/inner.vtree'))
    # return alpha


# build_inner_constraint("data")

def build_tile_constraint(dim, save_path=None):
    sdd = SddManager(var_count=dim**2, auto_gc_and_minimize=True)
    alpha = None
    for i in range(dim):
        for j in range(dim):
            if i == 0 or i == dim-1 or j == 0 or j == dim-1:
                nbrs = get_neighbors(i,j,dim)
                clause = None
                clause.ref()

            else:
                nbrs = get_neighbors(i,j,dim)
                clause = None
                clause.ref()
                for a in range(len(nbrs)-1):
                    for b in range(a+1,len(nbrs)):
                        off_clause = conjunction([-sdd.vars[get_idx(nbrs[k][0],nbrs[k][1],dim)] for k in range(len(nbrs)) if (k != a and k != b)])
                        off_clause.ref()
                        cur_clause = sdd.vars[get_idx(nbrs[a][0],nbrs[a][1],dim)] & sdd.vars[get_idx(nbrs[b][0],nbrs[b][1],dim)] \
                            & off_clause
                        if clause is None:
                            clause = cur_clause
                            clause.ref()
                        else:
                            prev_clause = clause
                            clause = clause | cur_clause
                            clause.ref()
                            prev_clause.deref()
                        cur_clause.deref()
                        off_clause.deref()
                cur_alpha = -sdd.vars[get_idx(i,j,dim)] | clause
                if alpha is None:
                    alpha = cur_alpha
                    alpha.ref()
                else:
                    prev_alpha = alpha
                    alpha = alpha & cur_alpha
                    alpha.ref()
                    prev_alpha.deref()
                cur_alpha.deref()
    print(f"Model Count: {alpha.wmc(log_mode=False).propagate()}")
    if save_path is not None:
        alpha.save(str.encode(f'{save_path}/{dim}x{dim}.sdd'))
        alpha.vtree().save(str.encode(f'{save_path}/{dim}x{dim}.vtree'))
    return alpha