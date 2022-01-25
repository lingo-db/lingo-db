# event_analyzing_sample.py: general event handler in python
# SPDX-License-Identifier: GPL-2.0
#
# Current perf report is already very powerful with the annotation integrated,
# and this script is not trying to be as powerful as perf report, but
# providing end user/developer a flexible way to analyze the events other
# than trace points.
#
# The 2 database related functions in this script just show how to gather
# the basic information, and users can modify and write their own functions
# according to their specific requirement.
#
# The first function "show_general_events" just does a basic grouping for all
# generic events with the help of sqlite, and the 2nd one "show_pebs_ll" is
# for a x86 HW PMU event: PEBS with load latency data.
#

from __future__ import print_function

import os
import sys
import math
import struct
import json

sys.path.append(os.environ['PERF_EXEC_PATH'] + \
                '/scripts/python/Perf-Trace-Util/lib/Perf/Trace')

from EventClass import *
from perf_trace_context import perf_sample_srcline

#
# If the perf.data has a big number of samples, then the insert operation
# will be very time consuming (about 10+ minutes for 10000 samples) if the
# .db database is on disk. Move the .db file to RAM based FS to speedup
# the handling, which will cut the time down to several seconds.
#

print(perf_sample_srcline)

events=[]


def trace_begin():
    pass


def add_jit_loc(info, obj):
    if "jit_loc" in obj:
        return
    if "dso" in info and "llvm-jit-static" in info["dso"]:
        obj["jit_loc"] = {}
        if "srcline" in info:
            obj["jit_loc"]["srcline"] = info["srcline"]
        if "symbol" in info:
            obj["jit_loc"]["symbol"] = info["symbol"]


def add_rt_loc(info, obj):
    if "rt_loc" in obj:
        return
    if "symbol" in info and info["symbol"].startswith("rt_"):
        obj["rt_loc"] = {}
        obj["rt_loc"]["symbol"] = info["symbol"]
        if "srcline" in info and ".mlir" in info["srcline"]:
            obj["rt_loc"]["srcline"] = info["srcline"]


def add_event_loc(info, obj):
    if "event_loc" in obj:
        return
    obj["event_loc"] = {}
    if "symbol" in info:
        obj["event_loc"]["symbol"] = info["symbol"]
    if "srcline" in info:
        obj["event_loc"]["srcline"] = info["srcline"]


def called_from_generated(callchain):
    generated_srcline = None
    for cf in callchain:
        if "srcline" in cf and ".mlir" in cf["scrline"]:
            generated_srcline = "generated_srcline"
    return generated_srcline


def process_event(param_dict):
    if param_dict["comm"] != "db-run-query":
        return
    event_dict = {"event": param_dict["ev_name"], "time": param_dict["sample"]["time"]}
    add_event_loc(param_dict, event_dict)
    add_jit_loc(param_dict, event_dict)
    add_rt_loc(param_dict, event_dict)

    if "callchain" in param_dict and len(param_dict["callchain"]) > 0:
        for cf in param_dict["callchain"]:
            add_jit_loc(cf, event_dict)
            add_rt_loc(cf, event_dict)
    #if "jit_loc" not in event_dict:
    #print(param_dict)
    events.append(event_dict)

def trace_end():
    with open('perf-result.json', 'w') as f:
        json.dump(events, f)


def trace_unhandled(event_name, context, event_fields_dict):
    print(' '.join(['%s=%s' % (k, str(v)) for k, v in sorted(event_fields_dict.items())]))
