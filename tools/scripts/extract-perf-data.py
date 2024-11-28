from __future__ import print_function

import os
import sys






sys.path.append(os.environ['PERF_EXEC_PATH'] + \
                '/scripts/python/Perf-Trace-Util/lib/Perf/Trace')


aggregated_generated = {

}
aggregated_rt_symbols = {

}
total_events=0
generated_events=0

def trace_begin():
    pass



def process_event(param_dict):
    global total_events
    global generated_events
    srcline = param_dict["srcline"] if "srcline" in param_dict else None
    symbol = param_dict["symbol"] if "symbol" in param_dict else None
    dso = (param_dict["dso"].split("/")[-1] if "dso" in param_dict else None)
    loc_type = None
    if srcline is not None:
        if ".mlir" in srcline:
            if srcline not in aggregated_generated:
                aggregated_generated[srcline]=0
            aggregated_generated[srcline]+=1
            generated_events+=1
    full_symbol= (dso if dso is not None else "")+":"+(symbol if symbol is not None else "")
    if full_symbol not in aggregated_rt_symbols:
        aggregated_rt_symbols[full_symbol]=0
    aggregated_rt_symbols[full_symbol]+=1
    total_events+=1



def trace_end():
    print(aggregated_generated)
    print(aggregated_rt_symbols)
    print(total_events)
    print(generated_events)


def trace_unhandled(event_name, context, event_fields_dict):
    print(' '.join(['%s=%s' % (k, str(v)) for k, v in sorted(event_fields_dict.items())]))
