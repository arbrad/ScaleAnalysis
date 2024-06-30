# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:52:21 2023

@author: Aaron
"""

from lib import VoronoiTessellation
from matplotlib import pyplot as plt, animation
from matplotlib.collections import LineCollection
import numpy as np
import random
from scipy.ndimage import convolve, maximum_filter, generate_binary_structure, gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats.qmc import PoissonDisk
import shapely

def runit(): return random.uniform(0, 1)

def dist(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)

def random_unit(N, mult=1):
    points = set((runit(), runit()) for _ in range(int(N * mult)))
    while len(points) > N:
        mindist_point = list(reversed(sorted(
            (min((dist(p, q), q) 
                 for q in points if p != q), p) 
            for p in points)))
        mindist_point.sort()
        mindist_point.reverse()
        while len(points) > N:
            (_, q), p = mindist_point.pop()
            if q not in points: break
            points.remove(p)
    return list(points)

def poisson_disk(N):
    engine = PoissonDisk(d=2, radius=0.05)
    return engine.random(N)

def plot(points, radius=0.25):
    vor = Voronoi(points)
    voronoi_plot_2d(vor)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    vtess = VoronoiTessellation(points, shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), 
                                keep=((0.5, 0.5), radius))
    print(len(vtess.polygons))
    vtmetrics, _ = vtess.tessellation_metrics()
    print(vtmetrics)
    dmetrics, _ = vtess.distribution_metrics()
    print(dmetrics)
    
def laplace_op(gamma=1/3):
    a = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    b = np.array([[1/2, 0, 1/2], [0, -2, 0], [1/2, 0, 1/2]])
    return (1-gamma)*a + gamma*b

ani = None
def save():
    if ani is not None:
        ani.save('rd.gif')
def pause():
    if ani is not None:
        ani.pause()
def resume():
    if ani is not None:
        ani.resume()

def schnakenberg(a=0.25,    # rate of supply of u
                 b=0.75,    # rate of supply of v
                 Du=0.195,  # diffusion coefficient of u
                 Dv=3.961,  # diffusion coefficient of v
                 dx=0.05,   # spatial discretization
                 dt=0.01,   # temporal discretization
                 r0=0.05,   # noise for initial concentrations
                 r1=0.05,   # percent noise to apply to a, b, Du, Dv
                 ss=3,      # noise smoothing, higher is smoother
                 steps=100):
    Lx, Ly = 5, 5  # unit square
    Nx, Ny = int(Lx/dx), int(Ly/dx)

    def rand():
        return np.random.rand(Nx, Ny)
    def smooth():
        noise = np.random.normal(0, 1, (Nx, Ny))
        return gaussian_filter(noise, sigma=ss, mode='wrap')

    a = a + a * r1 * smooth()
    b = b + b * r1 * smooth()
    Du = Du + Du * r1 * smooth()
    Dv = Dv + Dv * r1 * smooth()
    
    u = np.ones((Nx, Ny)) + r0 * smooth()
    v = np.zeros((Nx, Ny)) + r0 * smooth()
    
    # Laplacian
    kernel = laplace_op()
    
    fig, ax = plt.subplots()
    img = ax.imshow(u, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_aspect('equal')
    plt.colorbar(img, ax=ax)

    neighborhood = generate_binary_structure(2, 2)  # Defines neighborhood for comparison

    text = None
    def update(frame):
        nonlocal u, v, text
        #print([round(np.linalg.norm(x), 2) for x in [u, v]])
        
        for _ in range(steps):
            # Perform the simulation step (same as in your loop)
            # Compute Laplacian
            laplacian_u = convolve(u, kernel, mode='wrap')
            laplacian_v = convolve(v, kernel, mode='wrap')
        
        
            # Update concentrations
            u += dt * (a - u + u**2 * v + Du * dt * laplacian_u / dx**2)
            v += dt * (b - u**2 * v + Dv * dt * laplacian_v / dx**2)
            
    
        # Find local maxima
        local_max = (u == maximum_filter(u, footprint=neighborhood))  # Find local maxima
        
        # Redraw the heatmap
        img = ax.imshow(u, cmap='hot', interpolation='nearest', vmin=0, vmax=1)

        # Overlay dots at local maxima
        maxima_y, maxima_x = np.where(local_max)
        dots, = ax.plot(maxima_x, maxima_y, 'bo')  # 'bo' plots blue dots

        maxima_coordinates = list(zip(maxima_x, maxima_y))
        vtess = VoronoiTessellation(maxima_coordinates, shapely.Polygon([(0, 0), (Nx, 0), (Nx, Ny), (0, Ny), (0, 0)]), 
                                    keep=((Nx/2, Ny/2), 0.4*Nx))
        stm, tmetrics = vtess.tessellation_metrics()
        dtm, dmetrics = vtess.distribution_metrics()
        s = stm + '\n' + dtm
        # if text is None:
        #     text = ax.text(0.02, 0.02, s, transform=ax.transAxes, fontsize=9, 
        #                    bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 2},
        #                    family='monospace')
        # text.set_text(s)
        print(s)

        segments = ax.add_collection(LineCollection(vtess.edges()))

        return [img, dots, segments]

    global ani
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=50, blit=True)
    plt.show()


def random_voronoi(N=100):
    ps, bs, rps, rbs, reg = [], [], [], [], []
    while len(ps) < N:
        generators = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(100)]
        generators = np.array(generators)
        vtess = VoronoiTessellation(generators)
        if len(list(vtess._components())) > 1:
            continue
        # generators, _ = vtess.inverse_rate_voronoi()
        # s, _ = vtess.voronoi_metrics(generators)
        # print(s)
        _, (p, b) = vtess.voronoi_metrics()
        ps.append(p)
        bs.append(b)
        _, (p, b) = vtess.voronoi_metrics(vtess.random_generators())
        rps.append(p)
        rbs.append(b)
        reg.append(vtess.chi())
        # print(vtess.chi())
    ps = np.array(ps)
    bs = np.array(bs)
    rps = np.array(rps)
    rbs = np.array(rbs)
    reg = np.array(reg)
    print('perp', np.mean(ps), np.std(ps))
    print('bis', np.mean(bs), np.std(bs))
    print('rperp', np.mean(rps), np.std(rps))
    print('rbis', np.mean(rbs), np.std(rbs))
    print('chi', np.mean(reg), np.std(reg))