import os
import  subprocess
import sys

Lx = int(sys.argv[1])
Ly = int(sys.argv[2])

Us      = [2, 4, 8]
mus     = {
    2 : [0.2 * i for i in range(10)],
    4 : [0.5 * i for i in range(10)],
    8 : [0.5 * i for i in range(16)]
}
betas   = [4, 2, 1]
dtau    = 0.05
M0      = 10
outp    =   open(f"output{Ns}.dat", "a+")

for beta in betas:
    for U in Us:
        for mu in mus[U]:
            result = subprocess.run(f"sh skrypt_run_simulation.sh {Lx} {Ly} {dtau} {beta} {M0} {U} {mu} Sweep",  stdout=subprocess.PIPE)
            print(result)
