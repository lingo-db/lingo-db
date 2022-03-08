from profile_data import createOpNameFromRepr

def memory_heatmaps(data):
    relevant_ops = data.getOperations(cols=["id", "repr"], level=0, nested_inside="func")
    figs = []
    for op in relevant_ops:
        opId = op[0]
        opName = createOpNameFromRepr(op[1])
        data.con.execute("""select  e.virt_addr as addr,e.time as t 
                            from event e, operation op, operation op1, operation op2, operation op3 
                            where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline and (op.id=? or op.parent=? ) and e.virt_addr is not null
                            """, [opId, opId])

        mem_accesses = data.con.fetchdf()

        import numpy as np
        from scipy import stats

        mem_accesses = mem_accesses[(np.abs(stats.zscore(mem_accesses['addr'])) < 2.5)]
        mem_accesses['dA'] = mem_accesses['addr'].shift(1) - mem_accesses['addr'].shift(0)
        import plotly.express as px

        fig = px.density_heatmap(mem_accesses, x='t', y='dA', nbinsx=150, nbinsy=150, color_continuous_scale='YlGnBu_r')
        figs.append((opId,opName,fig))
    return figs