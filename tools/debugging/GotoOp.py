## > IRDUMPINSTRUCTIONID=1 SKIPQUERYEXECUTION=1 BUFFERSIZE=4096 DUMPMODULE=module.txt PARALLEL=1 rr record ./bin/sql sf1 <(cat <(echo '\o -') scripts/tpch/queries/16.sql)
## > rr replay -- --command tools/gdb/GotoInstruction.py
## # Navigate to the instruction with id 673:
## > goto-instruction 673
## > bt
##---------------------------------------------------------------------------

import gdb
import os
cwd=os.getcwd()
print(cwd)






class Cont:
    def __call__(self):
        gdb.execute("continue")


def continueAfterRestart(event):
    gdb.events.stop.disconnect(continueAfterRestart)
    gdb.post_event(Cont())


class Restart:
    def __init__(self, line):
        self.line = line

    def __call__(self):
        gdb.execute("set $suppress_run_hook = 1")
        # Set conditional breakpoint so that we stop at the correct line
        gdb.execute("tb runner.cpp:224 if operationId == " + str(self.line))  # sets a temporary breakpoint (gets removed when first hit)
        gdb.events.stop.connect(continueAfterRestart)
        # Start execution from beginning
        gdb.execute("run")
        gdb.execute("set $suppress_run_hook = 0")

    def stop_handler(self, event):
        gdb.execute("continue")


def runGotoInstr(instr):
    gdb.post_event(Restart(instr))




class GotoInstruction(gdb.Command):
    def __init__(self):
        super(GotoInstruction, self).__init__("goto-op", gdb.COMMAND_DATA)
    def matches(self, frame):
        try:
            filename=frame.function().symtab.fullname()
            if filename.startswith(cwd+"/lib") and not filename.endswith("runner.cpp"):
                print(filename)
                print(filename)
                return True
            else:
                return False
        except BaseException as err:
            return False
    
    def invoke(self, arg, from_tty):
        gdb.execute("set $suppress_run_hook = 1")
        # Set conditional breakpoint so that we stop at the correct line
        gdb.execute("tb runner.cpp:224 if operationId == " + arg)  # sets a temporary breakpoint (gets removed when first hit)
        # Start execution from beginning
        gdb.execute("run")
        gdb.execute("continue")
        gdb.execute("set $suppress_run_hook = 0")
        # Select lowest call frame that belongs to any translator
        frame = gdb.newest_frame()
        while frame:
            if self.matches(frame):
                frame.select()
                break
            frame = frame.older()




# This registers the goto instruction command with the gdb runtime
GotoInstruction()

