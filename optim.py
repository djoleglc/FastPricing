import scipy
import scipy.optimize


def optim(obj_func, initial_theta, bounds):
    optimResult = scipy.optimize.minimize(
        obj_func, initial_theta, method="COBYLA", jac=False
    )
    theta_opt = optimResult.x
    func_min = optimResult.fun
    return theta_opt, func_min
