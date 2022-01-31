import pygments as pg
from pygments.lexers import LlvmLexer
from pygments.formatters import HtmlFormatter
import dash_dangerously_set_inner_html
import plotly as pl
from dash import dcc, html


class PerfHtmlFormatter(HtmlFormatter):
    def __init__(self, percentages, color_map=None):
        self.percentages = percentages
        self.color_map = color_map
        super().__init__()

    def get_pfmt(self, line):
        if line in self.percentages:
            return "{:2.1f}%".format(self.percentages[line]).rjust(4)
        return "    "

    def get_p(self, line):
        return self.percentages[line] if line in self.percentages else 0

    def wrap(self, source, outfile):
        return self._wrap_code(source)

    def _wrap_code(self, source):
        yield 0, '<div class="highlight" style="width:100%;height:90vh;overflow-x:hidden;overflow-y:scroll;white-space:nowrap;background-color:rgb(255, 245, 235);">'
        yield 0, '<code>'
        line = 1
        for i, t in source:
            if i == 1:
                p = self.get_p(line)
                color = "0xffffff"
                if self.color_map is None:
                    color = pl.colors.sample_colorscale(pl.colors.sequential.Oranges, [sqrt(p / 100)], low=0.0,
                                                        high=1.0, colortype='rgb')[0]
                else:
                    color = self.color_map[line] if line in self.color_map else color

                yield 0, '<span style="display: inline-block;width:50px">' + self.get_pfmt(line) + '</span>'
                yield 0, '<span style="background-color:' + color + ';">'

                # it's a line of formatted code

                t += '<br>'
                yield 1, t
                yield 0, '</span>'
                line += 1
            else:
                yield i, t
        yield 0, '</code>'
        yield 0, '</div>'


def createPerfCodeView(data, level, colorLineMap):
    if level != 3:
        raise NotImplementedError()
    data.con.execute(
        """select count(*) as cnt
           from event e""")
    total_events = data.con.fetchone()[0]
    data.con.execute(
        """select op.loc, count(*) as cnt
           from operation op, event e
           where op.loc=e.jit_srcline group by op.loc order by cnt desc""")

    percentages = {}
    for ele in data.con.fetchall():
        percentages[int(ele[0].split(":")[1])] = (ele[1] * 100) / float(total_events)

    code = pg.highlight(data.sourcefiles[level], LlvmLexer(), PerfHtmlFormatter(percentages, colorLineMap))

    return html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(code))


def createColorLineMap(data, level, colomap):
    if level != 3:
        raise NotImplementedError()
    res = {}
    data.con.execute(
        """select op.id, op.parent,op3.loc as cnt
           from operation op,operation op1,operation op2,operation op3
           where op3.mapping=op2.id and op2.mapping=op1.id and op1.mapping=op.id""")
    for o in data.con.fetchall():
        c_id = colomap.lookupRGBA(o[0], 0.3)
        c_pid = colomap.lookupRGBA(o[1] or -1, 0.3)
        color = c_id or c_pid
        if color is not None:
            line = int(o[2].split(":")[1])
            res[line] = color
    return res
