import scipy
import scipy.optimize

def optim(self, obj_func, initial_theta, bounds):
        """
        Inputs:
            obj_func : function
                     function to minimize
            initial_theta : float or list of float
            bounds : list of tuples
        Outputs:
            theta_opt : float
            func_min : float
        """

        optimResult = scipy.optimize.minimize(
            obj_func, initial_theta, method="COBYLA", jac=False
        )
        theta_opt = optimResult.x
        func_min = optimResult.fun
        return theta_opt, func_min
