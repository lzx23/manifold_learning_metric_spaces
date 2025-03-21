import numpy as np

def eps_ball(dims, pos, eps):
    '''
    dims : tuple, shape of each image
    pos : np array of shape (num_samples, len(dims)), locations of centers
    eps : radius of balls in pixels

    Returns:
    res: np array of shape (num_samples,) + dims; stack of images of balls of radius eps centered at pos
    '''
    grid = np.indices(dims)
    grid = np.moveaxis(grid, 0, -1) # grid is now of shape dims + (len(dims),)

    res = np.empty((pos.shape[0],) + dims)

    for i, p in enumerate(pos):
        temp = np.where(np.linalg.norm(grid - p, axis = -1) < eps, 1, 0)
        res[i] = temp

    return res

def dtheta(th1, th2):
    '''
    th1, th2: angles in radians

    Returns:
    th1-th2 mod 2pi (the result will lie in [0, 2pi))
    '''
            
    dth = np.abs(th1 - th2)
    return np.minimum(dth, 2*np.pi - dth)

def r2_norm(th1, th2):
    '''
    th1, th2: angles in radians

    Returns:
    R^2 norm between (cos th1, sin th1) and (cos th2, sin th2)
    '''
    return np.sqrt((np.cos(th1) - np.cos(th2))**2 + (np.sin(th1) - np.sin(th2))**2)

def gen_data_s1(num_samples, L, eps, thetas = None, seed = 12345):
    '''
    Generates data for the rotating ball example

    num_samples: number of samples, only used if thetas is None
    L : length of image
    eps : radius of ball in pixels
    thetas : list or np array, angles which determine the centers

    Returns:
    res: np array of shape (num_samples,) + dims
    thetas : angles corresponding to the centers
    
    Note: if you plot the images with imshow, everything is rotated by pi/2, i.e., the image for theta = 0 looks like a ball centered at (0, 1)
        (but this doesn't matter due to symmetry)
    '''
    if thetas is None:
        rng = np.random.default_rng(seed = seed)
        thetas = 2*np.pi*rng.random((num_samples,))

    pos = np.array([[.5*(L-1) + np.cos(t)*.25*L, .5*(L-1) + np.sin(t)*.25*L] for t in thetas])
    return eps_ball((L, L), pos, eps), thetas