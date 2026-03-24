from psychopy import parallel, core

port = parallel.ParallelPort(address=0x3FD8)

print("Sending trigger")
port.setData(255)

core.wait(1)

port.setData(0)
print("Done")