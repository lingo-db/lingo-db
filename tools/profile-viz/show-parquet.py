import pyarrow as pa
import pyarrow.parquet as pq

table2 = pq.read_table('/home/michael/projects/code/events.parquet')
print(table2)

import dash
import dash_html_components as html


