import scipy 
import scipy.optimize

def optim(obj_func, initial_theta, bounds):
    opt = {}
    opt["maxiter"] = 5000
    optimResult = scipy.optimize.minimize(
        obj_func, initial_theta, tol=1e-20, method="COBYLA", jac=False, options=opt
    )
    theta_opt = optimResult.x
    func_min = optimResult.fun
    return theta_opt, func_min
