import pp
import subprocess
import numpy as np
import time

def start_workers(N, template = "submit.template"):
    # assume sbatch and ppserver.py in path
    f = " . "
    if N >= 1:
        f = subprocess.check_output(['sbatch','--array=1-%d' % N, template])
    # server, jobid
    return f.split()[-1]

def start_server(local_cpus = 0):
    #ppservers = ("*",) # auto-discover
    ppservers = ("*",) # auto-discover
    job_server = pp.Server(local_cpus, ppservers=ppservers)
    return job_server

def kill_workers(jobid):
    subprocess.call(['scancel',jobid])


#    f = job_server.submit(proc.t_log, (ad_x, ))
#
#for x, f in results:
#    # Retrieves the result of the calculation
#    val = f()
#    print "t_log(%lf) = %lf, t_log'(%lf) = %lf" % (x, val.x, x, val.dx)

#jobs = [(input, job_server.submit(sum_primes, (input, ), (isprime, ),
#        ("math", ))) for input in inputs]
#
#for input, job in jobs:
#    print "Sum of primes below", input, "is", job()

#job_server.wait()


# Server.destroy(self)
# _Task().self.finished
