"""
    Simulazioni per il sistema elastico e oscillatori con la stessa frequenza.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import odeint
from scipy.optimize import dual_annealing, minimize, OptimizeResult

def simulate_damped(type, t, y_damped, omega, gamma, x0, N_MAX = 21):
    """According to type ("elastic_one", "elastic_many", "cutoff_one",
    "cutoff_many") performs the optimization against the y_damped curve,
    in the t time interval, being omega, gamma and x0 the parameters and
    initial displacement of the damped motion (initial velocity is zero,
    masses are 1). Performs the simulations 3,5,...,N_MAX degrees of freedom:
    the result is a list of triples [N0, parameters, y] where parameters are
    the optimized parameters and y is the resulting approximating curve."""
    
    yy = []
    previous_loss = 1e20
    for N0 in range(1, (N_MAX-1)//2+1):
        print()
        N = 2*N0 + 1
        NN = 2*N

        # Choose one of the following functions/bounds according to the type
        # parameter: since we use N and NN inside the equation() functions
        # we need to define them here because of Python lexical scoping rules.
        if type == "elastic_one":
            def equation(y, t, *params):
                """Return the NN second members of the 1st order ODE system
                which implements the simple elastic coupling, where all
                oscillator frequencies are equal to omega**2 == params[0]
                and the coupling constants are params[1], ..."""
                e = np.zeros((NN,))
                for i in range(N):
                    e[i] = y[N+i]
                e[N] = params[1] * y[1] - params[0] * y[0]
                for i in range(1, N-1):
                    e[N+i] = params[i] * y[i-1] + params[i+1] * y[i+1] - params[0] * y[i]
                e[NN-1] = params[-1] * y[N-2] - params[0] * y[N-1]
                return e
            bounds = [(omega**2/10,omega**2*10)] + [(2*gamma/10,2*gamma*10)] *(N-1)
        elif type == "elastic_many":
            def equation(y, t, *params):
                """Returns the NN second members of the 1st order ODE system
                which implements the simple elastic coupling, where oscillator
                frequencies are params[0], ... and the coupling constants
                params[N], ..."""
                e = np.zeros((NN,))
                for i in range(N):
                    e[i] = y[N+i]
                e[N] = params[N] * y[1] - params[0] * y[0]
                for i in range(1, N-1):
                    e[N+i] = params[N+i-1] * y[i-1] + params[N+i] * y[i+1] - params[i] * y[i]
                e[NN-1] = params[-1] * y[N-2] - params[N-1] * y[N-1]
                return e
            bounds = ([(omega**2/10,omega**2*10)] *N) + [(2*gamma/10,2*gamma*10)] *(N-1)
        elif type == "cutoff_one":
            def equation(y, t, *params):
                """Returns the NN second members of the 1st order ODE system
                which implements the cutoff Schrodinger coupling, where
                oscillator frequencies is params[0] and the coupling constants
                params[1], ..."""
                e = np.zeros((NN,))
                for i in range(N):
                    e[i] = y[N+i]
                e[N] = params[0] * (y[1] - y[0])
                for i in range(1, N-1):
                    e[N+i] = params[0] * (y[i+1] - y[i]) + params[0] * (y[i-1] - y[i])
                e[NN-1] = params[0] * (y[N-2] - y[N-1]) - params[0] * y[N-1]
                return e
            bounds = [(omega**2/10,omega**2*10)]
        elif type == "cutoff_many":
            def equation(y, t, *params):
                e = np.zeros((NN,))
                for i in range(N):
                    e[i] = y[N+i]
                e[N] = params[0] * (y[1] - y[0])
                for i in range(1, N-1):
                    e[N+i] = params[i] * (y[i+1] - y[i]) + params[i+1] * (y[i-1] - y[i])
                e[NN-1] = params[-2] * (y[N-2] - y[N-1]) - params[-1] * y[N-1]
                return e
            bounds = [(omega**2/10,omega**2*10)] * N
        else:
            raise "Bad type parameter"

        print(N, "degrees of freedom optimization with bounds", bounds)

        # Initial conditions to solve the system
        init_cond = np.zeros((NN,))
        init_cond[N0] = x0

        def loss(params):
            y = odeint(equation, init_cond, t, args = tuple(params))[:,N0]
            return np.linalg.norm(y - y_damped, ord=np.inf)  # sup norm

        def annealing_callback(x, f, context):
            if context == 0:
                print("... Minimum found! Loss:", loss(x))
            return False

        m_best = OptimizeResult()
        l_best = 1e20
        for i in range(10):
            # Tries ten times to improve the result of the previous N0
            print(f"Optimization trials {i}/10")
            m = dual_annealing(loss, bounds, 
                maxiter = 100,
                callback = annealing_callback)
            # Compare with previous loss: repeat if no improvement
            l = loss(m.x)
            if l < l_best:
                l_best = l
                m_best = m
            if l < previous_loss:
                previous_loss = l
                break

        optima = m_best.x
        y = odeint(equation, init_cond, t, args = tuple(optima))[:,N0]
        yy.append((N0, optima, y))
        print()
        print("Minimization", "ok." if m.status else "KO!", m.message)
        print("Loss: ", loss(optima))
        print("Result [omega_k**2, omega_{hk}]:", optima)
    return yy

def result_plot(t, y_damped, yy, filename = None):
    """Plot the list yy of results from an optimization against a damped
    curve. Saves the plot on files whose names stem from filename: if filename
    is omitted then picture are shown."""
    plt.figure(figsize=(10, 5))
    plt.grid()
    for record in yy:
        plt.plot(t,record[2], label=f"$N={record[0]}$")
    plt.plot(t,y_damped, 'black', label="damped")
    plt.legend()
    if filename != None:
        plt.savefig(filename + ".png")
    else:
        plt.show()
    
    plt.figure(figsize=(10, 5))
    ascisse = np.arange(yy[0][0]*2+1,yy[-1][0]*2+2,2)
    errori = [np.linalg.norm(y_damped - np.array(yy[i][2]), ord=np.inf) for i in range(len(yy))]
    plt.xticks(ascisse)
    plt.yticks(np.linspace(0,errori[0],len(errori)))
    plt.grid()
    plt.plot(ascisse,errori,'-o')
    if filename != None:
        plt.savefig(filename + "_error.png")
    else:
        plt.show()
    
    # Sceglie l'errore minimo e mostra gli omega corrispondenti
    imin = 0
    errore = 1e20
    for i in range(len(errori)-1, -1, -1):
        if errori[i] < errore:
            errore = errori[i]
            imin = i
    N = (imin+1)*2+1
    print(f"Best approximation for N = {N} (loss = {errore:.4})")
    print(f"omega = {np.sqrt(yy[imin][1][0]):.4}")
    
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(t,yy[-1][2], label=f"$N={yy[-1][0]}$")
    plt.plot(t,y_damped, 'black', label="damped")
    plt.legend()
    if filename != None:
        plt.savefig(filename + "_single.png")
    else:
        plt.show()

t = np.linspace(0, 5, 100)  # Time interval sample considered

# Parameters of the damped system
gamma = 0.5
omega = 4
x0 = 1
p0 = 0
a = 2*gamma
k = omega*omega

# Computes the solution y_damped of x''+gamma x' + omega^2 x = 0
y_damped = odeint(lambda y, t: [y[1], -2*a*y[1] - k*y[0]], [x0, p0], t)[:,0]

# Computes the solution y_damped2 of x''+gamma x'|x'| + omega^2 sin x = 0
y_damped2 = odeint(lambda y, t: [y[1], -a*y[1]*abs(y[1]) - k*np.sin(y[0])], [x0, p0], t)[:,0]

# Just uncomment one of the following instruction groups to perform
# a specific simulation

print("Elastic coupling, same frequencies, simple damping.")
yy = simulate_damped("elastic_one", t, y_damped, omega, gamma, x0)
result_plot(t, y_damped, yy, "elastic_one_damped")
print(yy)

#~ print("Elastic coupling, different frequencies, simple damping.")
#~ yy = simulate_damped("elastic_many", t, y_damped, omega, gamma, x0)
#~ result_plot(t, y_damped, yy, "elastic_multi_damped")
#~ print(yy)

#~ print("Elastic coupling, same frequencies, quadratic damping.")
#~ yy = simulate_damped("elastic_one", t, y_damped2, omega, gamma, x0)
#~ result_plot(t, y_damped2, yy, "elastic_one_damped2")
#~ print(yy)

#~ print("Elastic coupling, different frequencies, quadratic damping.")
#~ yy = simulate_damped("elastic_many", t, y_damped2, omega, gamma, x0)
#~ result_plot(t, y_damped2, yy, "elastic_many_damped2")
#~ print(yy)

#~ print("Cutoff coupling, same frequencies, simple damping.")
#~ yy = simulate_damped("cutoff_one", t, y_damped, omega, gamma, x0)
#~ result_plot(t, y_damped, yy, "cutoff_one_damped")
#~ print(yy)

#~ print("Cutoff coupling, different frequencies, simple damping.")
#~ yy = simulate_damped("cutoff_many", t, y_damped, omega, gamma, x0)
#~ result_plot(t, y_damped, yy, "cutoff_many_damped")
#~ print(yy)

#~ print("Cutoff coupling, same frequencies, quadratic damping.")
#~ yy = simulate_damped("cutoff_one", t, y_damped2, omega, gamma, x0)
#~ result_plot(t, y_damped2, yy, "cutoff_one_damped2")
#~ print(yy)

#~ print("Cutoff coupling, different frequencies, quadratic damping.")
#~ yy = simulate_damped("cutoff_many", t, y_damped2, omega, gamma, x0)
#~ result_plot(t, y_damped2, yy, "cutoff_many_damped2")
#~ print(yy)
