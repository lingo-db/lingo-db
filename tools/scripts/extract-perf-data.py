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

obj=[]
def get_or_add_obj(d,key,val):
	if key in d:
		return d[key]
	else:
		d[key]=val
		return d[key]



def trace_begin():
	pass


def process_event(param_dict):
	if param_dict["comm"]!="db-run-query":
		return
	print(param_dict)
	if "srcline" in param_dict:
		if ".mlir" in param_dict["srcline"]:
			obj.append({"srcline":param_dict["srcline"],"event":param_dict["ev_name"],"time":param_dict["sample"]["time"]})	
		elif "callchain" in param_dict and len(param_dict["callchain"])>0:
			last_ele=param_dict["callchain"][-1]
			if "dso" in last_ele and "jitted" in last_ele["dso"]:
				print(param_dict)






def trace_end():
	with open('perf-result.json', 'w') as f:
	   	json.dump(obj,f)



def trace_unhandled(event_name, context, event_fields_dict):
        print (' '.join(['%s=%s'%(k,str(v))for k,v in sorted(event_fields_dict.items())]))
