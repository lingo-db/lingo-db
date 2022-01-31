import json

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import re

operation_dict = {"id": [], "repr": [], "loc": [], "parent": [], "mapping": [], "level": [], "dependencies": []}


def get_or_none(obj, key):
    if obj is None: return None
    if key in obj:
        return obj[key]
    return None


with open("sourcemap.json", "r") as sourcemap_file:
    sourcemap = json.load(sourcemap_file)
    for op in sourcemap:
        operation_dict["id"].append(op["id"])
        operation_dict["repr"].append(op["representation"])
        operation_dict["loc"].append(op["loc"])
        operation_dict["parent"].append(get_or_none(op, "parent"))
        operation_dict["mapping"].append(get_or_none(op, "mapping"))
        operation_dict["dependencies"].append(get_or_none(op, "dependencies"))
        m = re.search(r'snapshot-(\d+).mlir.*', op["loc"])
        if m is not None:
            operation_dict["level"].append(int(m.group(1)))
        else:
            operation_dict["level"].append(-1)

    schema = pa.schema({
        "id": pa.uint64(),
        "repr": pa.string(),
        "loc": pa.string(),
        "parent": pa.uint64(),
        "mapping": pa.uint64(),
        "level": pa.int64(),
        "dependencies": pa.list_(pa.uint64())
    })
table = pa.Table.from_pydict(operation_dict, schema)
print(table)
pq.write_table(table, 'operations.parquet')
