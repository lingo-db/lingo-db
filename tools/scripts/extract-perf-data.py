from __future__ import print_function
import ctypes

import os
import sys



c_uint8 = ctypes.c_uint8
c_uint16 = ctypes.c_uint16
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64


class IBSData_1(ctypes.LittleEndianStructure):
    _pack = 1
    _fields_ = [
        ("comp_to_ret_ctr", c_uint16, 16),
        ("tag_to_ret_ctr", c_uint16, 16),
        ("reserved1", c_uint8, 2),
        ("op_return", c_uint8, 1),
        ("op_brn_taken", c_uint8, 1),
        ("op_brn_misp", c_uint8, 1),
        ("op_brn_ret", c_uint8, 1),
        ("op_rip_invalid", c_uint8, 1),
        ("op_brn_fuse", c_uint8, 1),
        ("op_microcode", c_uint8, 1),
        ("reserved2_1", c_uint8, 7),
        ("reserved2_2", c_uint16, 16),
    ]


class IBSData_2(ctypes.LittleEndianStructure):
    _fields_ = [
        ("data_src", c_uint8, 3),
        ("reserved0", c_uint8, 1),
        ("rmt_node", c_uint8, 1),
        ("cache_hit_st", c_uint8, 1),
        ("reserved1", c_uint64, 57),
    ]


class IBSData_3(ctypes.LittleEndianStructure):
    _pack = 1
    _fields_ = [
        ("ld_op", c_uint8, 1),
        ("st_op", c_uint8, 1),
        ("dc_l1tlb_miss", c_uint8, 1),
        ("dc_l2tlb_miss", c_uint8, 1),
        ("dc_l1tlb_hit_2m", c_uint8, 1),
        ("dc_l1tlb_hit_1g", c_uint8, 1),
        ("dc_l2tlb_hit_2m", c_uint8, 1),  # 8
        ("dc_miss", c_uint8, 1),
        ("dc_mis_acc", c_uint8, 1),
        ("reserved", c_uint8, 4),
        ("dc_wc_mem_acc", c_uint8, 1),
        ("dc_uc_mem_acc", c_uint8, 1),  # 16
        ("dc_locked_op", c_uint8, 1),
        ("dc_miss_no_mab_alloc", c_uint8, 1),
        ("dc_lin_addr_valid", c_uint8, 1),
        ("dc_phy_addr_valid", c_uint8, 1),
        ("dc_l2_tlb_hit_1g", c_uint8, 1),
        ("l2_miss", c_uint8, 1),
        ("sw_pf", c_uint8, 1),
        ("op_mem_width_1", c_uint8, 2),
        ("op_mem_width_2", c_uint8, 2),
        ("op_dc_miss_open_mem_reqs", c_uint8, 6),
        ("dc_miss_lat", c_uint16, 16),
        ("tlb_refill_lat", c_uint16, 16),
    ]


class IBSData(ctypes.Structure):
    _pack = 1
    _fields_ = [("op_ctl", c_uint64),
                ("rip", c_uint64),
                ("ibs_op_data", IBSData_1),
                ("ibs_op_data2", IBSData_2),
                ("ibs_op_data3", IBSData_3),
                ("virt_address", c_uint64),
                ("phys_address", c_uint64),
                ("br_target", c_uint64),
                ]


sys.path.append(os.environ['PERF_EXEC_PATH'] + \
                '/scripts/python/Perf-Trace-Util/lib/Perf/Trace')

import pyarrow as pa
import pyarrow.parquet as pq


events = {
    "time": [],
    "is_ld": [],
    "is_st": [],
    "l2_miss": [],
    "l3_miss": [],
    "virt_addr": [],
    "loc_type": [],  # None/jit/rt
    "rt_srcline": [],
    "symbol": [],
    "jit_srcline": []
}


def trace_begin():
    pass


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
    if param_dict["comm"] != "run-sql":
        return
    b = bytearray(param_dict["raw_buf"])
    ibs_data = IBSData.from_buffer_copy(b[4:])
    events["time"].append(param_dict["sample"]["time"])
    events["is_ld"].append(ibs_data.ibs_op_data3.ld_op == 1)
    events["is_st"].append(ibs_data.ibs_op_data3.st_op == 1)
    events["l2_miss"].append(ibs_data.ibs_op_data3.l2_miss == 1)
    events["l3_miss"].append(ibs_data.ibs_op_data3.dc_miss == 1)
    events["virt_addr"].append(ibs_data.virt_address if ibs_data.ibs_op_data3.dc_lin_addr_valid == 1 else None)
    srcline = param_dict["srcline"] if "srcline" in param_dict else None
    symbol = param_dict["symbol"] if "symbol" in param_dict else None
    loc_type = None
    if srcline is not None:
        if ".mlir" in srcline:
            loc_type = "jit"
        else:
            loc_type = "rt"
    if loc_type is None and symbol is not None:
        loc_type="rt"
    events["loc_type"].append(loc_type)
    rt_srcline = None
    jit_srcline = None
    if loc_type == "jit":
        jit_srcline = srcline
    if loc_type == "rt":
        rt_srcline = srcline
        line = int(param_dict["iregs"].strip().split(":")[1], base=16)
        if line < 0x80000000 and line >= 0:
            jit_srcline = "snapshot-3.mlir:" + str(line)
    events["rt_srcline"].append(rt_srcline)
    events["symbol"].append(symbol)
    events["jit_srcline"].append(jit_srcline)


def trace_end():
    schema = pa.schema({
        "time": pa.uint64(),
        "is_ld": pa.bool_(),
        "is_st": pa.bool_(),
        "l2_miss": pa.bool_(),
        "l3_miss": pa.bool_(),
        "loc_type":pa.string(),
        "rt_srcline": pa.string(),
        "virt_addr": pa.uint64(),
        "symbol": pa.string(),
        "jit_srcline": pa.string(),
    })
    print(len(events["time"]))
    table = pa.Table.from_pydict(events, schema)
    pq.write_table(table, 'events.parquet')


def trace_unhandled(event_name, context, event_fields_dict):
    print(' '.join(['%s=%s' % (k, str(v)) for k, v in sorted(event_fields_dict.items())]))
