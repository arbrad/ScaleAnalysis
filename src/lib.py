# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:30:12 2023

@author: Aaron
"""

from collections import Counter, defaultdict
import copy
import cvxpy as cp
import math
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.spatial import Voronoi
import shapely
from sklearn.cluster import DBSCAN


# Many metrics are from https://kmh-lanl.hansonhub.com/uncertainty/meetings/gunz03vgr.pdf,
# compare values to p. 128.

# See also https://people.sc.fsu.edu/~jburkardt/publications/gb_2004.pdf


NORM = 1


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.adjacent = []

    def add_adjacent(self, vertex):
        self.adjacent.append(vertex)

    def coord(self):
        return self.x, self.y

    def move(self, x, y):
        self.x, self.y = x, y

    def contract_toward(self, C, u, rate):
        A = np.array(self.coord())
        P = C + u * np.dot(A-C, u)
        self.move(*tuple(A + rate * (P-A)))

    def sort_adjacent_polar(self):
        # Sort the adjacent vertices by their polar angle with respect to this vertex
        self.adjacent.sort(key=lambda v: math.atan2(v.y - self.y, v.x - self.x))
        self.adjacent = tuple(self.adjacent)
        
    def __str__(self):
        return f'({self.x}, {self.y})'
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    

class Polygon:
    def __init__(self, vertices, order=True):
        self.vertices = vertices
        self.centroid = self.calculate_centroid()
        if order:
            # BUG! Only works for convex polygons
            self.vertices = self.sort_vertices_counterclockwise()
            
    def update_centroid(self):
        self.centroid = self.calculate_centroid()

    def calculate_centroid(self):
        if not self.vertices:
            return None

        cx = 0.0
        cy = 0.0
        area = 0.0

        for i in range(len(self.vertices)):
            x1, y1 = self.vertices[i].coord()
            x2, y2 = self.vertices[(i + 1) % len(self.vertices)].coord()
            factor = x1 * y2 - x2 * y1
            cx += (x1 + x2) * factor
            cy += (y1 + y2) * factor
            area += factor

        area /= 2.0

        if area == 0.0:
            return None

        cx /= (6 * area)
        cy /= (6 * area)

        return (cx, cy)

    def sort_vertices_counterclockwise(self):
        if not self.centroid:
            return self.vertices

        return sorted(self.vertices, key=lambda vertex: math.atan2(vertex.y - self.centroid[1], vertex.x - self.centroid[0]))

    def calculate_perimeter(self):
        perimeter = 0.0
        n = len(self.vertices)

        for i in range(n):
            x1, y1 = self.vertices[i].coord()
            x2, y2 = self.vertices[(i + 1) % n].coord()
            edge_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            perimeter += edge_length

        return perimeter

    def calculate_area(self):
        area = 0.0
        n = len(self.vertices)

        for i in range(n):
            x1, y1 = self.vertices[i].coord()
            x2, y2 = self.vertices[(i + 1) % n].coord()
            factor = x1 * y2 - x2 * y1
            area += factor

        area /= 2.0
        return abs(area)

    def normalized_isoperimetric_ratio(self):
        perimeter = self.calculate_perimeter()
        area = self.calculate_area()
        return 4 * math.pi * area / (perimeter ** 2)

    def angular_defect(self):
        n = len(self.vertices)
        if n < 3:
            return None

        regular_angle = 2 * math.pi / n
        sum_diff = 0.0

        for i in range(n):
            x1, y1 = self.vertices[i].coord()
            x2, y2 = self.vertices[(i + 1) % n].coord()
            x3, y3 = self.vertices[(i + 2) % n].coord()

            vec1 = (x2 - x1, y2 - y1)
            vec2 = (x3 - x2, y3 - y2)

            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
            mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

            angle_diff = abs(math.acos(dot_product / (mag1 * mag2)) - regular_angle)
            sum_diff += angle_diff

        return 1 / (1 + sum_diff)
    
    def random_coord(self):
        A = self.vertices[0]
        tris = [[A] + self.vertices[i:i+2] for i in range(1, len(self.vertices)-1)]
        assert len(tris) == len(self.vertices) - 2
        weights = [Polygon(tri).calculate_area() for tri in tris]
        tri = random.choices(tris, weights=weights)[0]
        return sample_triangle(*tri)
        
    def max_diameter(self, axis):
        max_weighted_length = None
        max_diam = None
        for i in range(len(self.vertices)):
            s, t = self.vertices[i].coord()
            for j in range(i+1, len(self.vertices)):
                x, y = self.vertices[j].coord()
                A = np.array((s, t))
                B = np.array((x, y))
                AB = B - A
                l = np.linalg.norm(AB)
                u = AB / l
                crosses = [np.cross(np.array(v.coord()) - A, u) for v in self.vertices]
                ex0 = -min(crosses)
                ex1 = max(crosses)
                assert ex0 >= 0
                assert ex1 >= 0
                weighted = l/(ex0+ex1)
                if max_weighted_length is None or weighted > max_weighted_length:
                    max_weighted_length = weighted
                    max_diam = ((s, t), (x, y))
        (s, t), (x, y) = max_diam
        if s > x:
            s, t, x, y = x, y, s, t
        if x == s:
            angle = math.pi/2 if y > t else -math.pi/2
        else:
            angle = math.atan((y-t)/(x-s))
        if axis is not None:
            for i in range(len(axis)-1):
                if s >= axis[i][0] and s <= axis[i+1][0]:
                    A = np.array(axis[i])
                    B = np.array(axis[i+1])
                    AB = B - A
                    axis_angle = math.atan(AB[1]/AB[0])
                    angle -= axis_angle
                    if angle > math.pi/2: angle -= math.pi
                    if angle < -math.pi/2: angle += math.pi
        return angle, max_diam, max_weighted_length-1, self.centroid

    def max_rect(self, axis):
        def sdist(C, A, u):
            return np.cross(u, C-A)
        def perp(u):
            return np.array([-u[1], u[0]])
        nv = len(self.vertices)
        vs = [np.array(v.coord()) for v in self.vertices]
        max_ratio = None
        angle = None
        diam = None
        for i in range(nv):
            A = np.array(self.vertices[i].coord())
            B = np.array(self.vertices[(i+1) % nv].coord())
            AB = B-A
            u = AB / np.linalg.norm(AB)
            width = max(sdist(C, A, u) for C in vs)
            p = perp(u)
            pdists = [sdist(C, A, p) for C in vs]
            length = -min(pdists) + max(pdists)
            if length < width:
                length, width, u, p = width, length, p, u
            ratio = length/width
            if max_ratio is None or ratio > max_ratio:
                max_ratio = ratio
                if u[0] == 0:
                    angle = math.pi/2 if u[1] > 0 else -math.pi/2
                else:
                    angle = math.atan(u[1]/u[0])
                center = np.array(self.centroid)
                if axis is not None:
                    for i in range(len(axis)-1):
                        if center[0] >= axis[i][0] and center[0] <= axis[i+1][0]:
                            A = np.array(axis[i])
                            B = np.array(axis[i+1])
                            AB = B - A
                            axis_angle = math.atan(AB[1]/AB[0])
                            angle -= axis_angle
                            if angle > math.pi/2: angle -= math.pi
                            if angle < -math.pi/2: angle += math.pi
                diam = (tuple(center - u*length/2),
                        tuple(center + u*length/2))
        return angle, diam, max_ratio-1, self.centroid

    def aspect_ratio(self, axis):
        xy = np.array([v.coord() for v in self.vertices]).T
        center = np.array(self.centroid)
        eigvals, eigvecs = np.linalg.eig(np.cov(xy))
        maxi = int(eigvals[0] < eigvals[1])
        vec = eigvecs[maxi]
        val = eigvals[maxi]**0.5
        ratio = abs(max(eigvals)/min(eigvals))
        angle = math.atan(vec[1]/vec[0])
        diam = (tuple(center - vec*val),
                tuple(center + vec*val))
        return angle, diam, ratio-1, self.centroid

    def max_ratio_angle(self, axis):
        rv0 = self.max_diameter(axis)
        rv1 = self.max_rect(axis)
        assert rv1[2] >= rv0[2]
        if rv0[2] < rv1[2]:
            rv0, rv1 = rv1, rv0
        return rv0

    def ratio_along(self, angle):
        u = np.array((math.cos(angle), math.sin(angle)))
        p = np.array((-u[1], u[0]))
        vs = [np.array(v.coord()) for v in self.vertices]
        C = np.array(self.centroid)
        widths = [np.cross(u, V-C) for V in vs]
        width = -min(widths) + max(widths)
        lengths = [np.cross(p, V-C) for V in vs]
        length = -min(lengths) + max(lengths)
        return length/width

    def contract_toward(self, C, u, rate):
        poly = copy.deepcopy(self)
        for v in poly.vertices:
            v.contract_toward(C, u, rate)
        return poly

    def __len__(self):
        return len(self.vertices)
        

class Tessellation:
    def __init__(self, polygons):
        self.polygons = polygons
        self.external_generators = None    
        self.e2p = None
        self.p2i = None

    def region(self):
        return set_op(self.polygons, False)
    
    def _shapely_region(self):
        final = shapely_of(self.polygons[0])
        for poly in self.polygons[1:]:
            final = final.union(shapely_of(poly))
        return final
        
    def bounded_voronoi_of_centroids(self):
        boundary = self._shapely_region()
        generators = np.array(list(list(poly.centroid) for poly in self.polygons))
        return VoronoiTessellation(generators, boundary)

    def bounded_voronoi_of(self, generators):
        boundary = self._shapely_region()
        generators = np.array(list(list(g) for g in generators))
        return VoronoiTessellation(generators, boundary)
    
    def _standard_edge(self, edge):
        a, b = edge
        return tuple(sorted([a, b], key=lambda v: v.coord()))
    
    def _edge_key(self, polygon, i):
        v0, v1 = polygon.vertices[i], polygon.vertices[(i + 1) % len(polygon.vertices)]
        return self._standard_edge((v0, v1))

    def _edges_to_polygons(self):
        if self.e2p is None:
            e2p = defaultdict(list)
            for polygon in self.polygons:
                num_vertices = len(polygon.vertices)
                for i in range(num_vertices):
                    # Current edge is from vertex i to i+1, wrapping around to 0 at the end
                    edge = self._edge_key(polygon, i)
                    e2p[edge].append(polygon)
            self.e2p = e2p
        return self.e2p
    
    def _p2i(self):
        if self.p2i is None:
            # polygon to index
            self.p2i = {}
            for i, poly in enumerate(self.polygons):
                self.p2i[poly.centroid] = i
        return self.p2i

    def _mirror(self, A, B, C):
            # Compute vectors
            AB = B - A
            AC = C - A                
            # Unit vector along AB
            u = AB / np.linalg.norm(AB)
            # Projection of AC onto the line AB
            projection = A + np.dot(AC, u) * u
            # Compute vector PC and find D
            PC = C - projection
            D = projection - PC
            return D

    def _constraining_generators(self):
        if self.external_generators is None:
            # Deduce external centroids
            e2p = self._edges_to_polygons()
            self.external_generators = []
            for polygon in self.polygons:
                num_vertices = len(polygon.vertices)
                for i in range(num_vertices):
                    edge = self._edge_key(polygon, i)
                    if len(e2p[edge]) == 1:
                        # Deduce centroid D as vector reflection of centroid C across edge AB
                        A, B = [np.array(polygon.vertices[j].coord()) for j in (i, (i+1) % num_vertices)]
                        C = np.array(polygon.centroid)
                        # add as extra
                        self.external_generators.append(list(self._mirror(A, B, C)))
        return self.external_generators
    
    def _externals(self):
        e2p = self._edges_to_polygons()
        _e2p = {}  # different local representation
        v2e = defaultdict(list)
        groups = []
        e2g = {}
        # obtain external edges, oriented for external polygon
        for poly in self.polygons:
            for i in range(len(poly.vertices)):
                if len(e2p[self._edge_key(poly, i)]) != 1: continue
                A, B = [poly.vertices[j].coord() for j in (i, (i+1)%len(poly.vertices))]
                edge = (B, A)  # reverse b/c examining external polygon
                _e2p[edge] = poly
                v2e[A].append(edge)
                v2e[B].append(edge)
                e2g[edge] = len(groups)
                groups.append([edge])
        # form groups of convex adjacent edges
        for edges in v2e.values():
            e, f = edges
            if e[1] != f[0]: e, f = f, e
            A, B, C = [np.array(g) for g in (e[0], e[1], f[1])]
            AB = B - A
            BC = C - B            
            if AB[0]*BC[1] - AB[1]*BC[0] > 0:
                i, j = [e2g[g] for g in (e, f)]
                e2g[f] = i
                groups[i].extend(groups[j])
                groups[j].clear()
        # return with standardized edges (e.g., for use in e2p)
        def convert_edge(edge):
            (a, b), (c, d) = edge
            return (Vertex(a, b), Vertex(c, d))
        def convert(group):
            return [convert_edge(edge) for edge in group]            
        return [convert(group) for group in groups if len(group) > 0]

    def _external_generators(self, generators):
        p2i = self._p2i()
        e2p = self._edges_to_polygons()
        groups = self._externals()
        externals = []
        for i, group in enumerate(groups):
            P, p, D, d = [], [], [], []
            for edge in group:
                A, B = [np.array(v.coord()) for v in edge]
                AB = B - A
                u = AB / np.linalg.norm(AB)
                poly = e2p[self._standard_edge(edge)][0]
                j = p2i[poly.centroid]
                G = generators[j]
                prow = [0]*2
                prow[0] = u[0]
                prow[1] = u[1]
                P.append(prow)
                p.append(u[0]*G[0] + u[1]*G[1])
                drow = [0]*2
                drow[0] = u[1]
                drow[1] = -u[0]
                D.append(drow)
                d.append(-G[0]*u[1] + G[1]*u[0] + 2 * (A[0]*u[1] - A[1]*u[0]))
            P = np.array(P)
            D = np.array(D)
            x = cp.Variable(2)
            obj = cp.Minimize(cp.norm(P @ x - p, NORM)
                              + cp.norm(D @ x - d, NORM))
            prob = cp.Problem(obj, [])
            prob.solve(solver=cp.CLARABEL)
            externals.append(tuple(x.value))
        return externals

    def constrained_voronoi(self, generators=None):
        if generators is None:
            generators = [list(poly.centroid) for poly in self.polygons]
        generators = np.array(generators)
        extra = np.array(self._external_generators(generators))
        return VoronoiTessellation(generators, None, extra=extra)
    
    def edges(self):
        es = set()
        for poly in self.polygons:
            for i in range(len(poly.vertices)):
                a = poly.vertices[i].coord()
                b = poly.vertices[(i+1)%len(poly.vertices)].coord()
                if b[0] < a[0] or b[0] == a[0] and b[1] < a[1]:
                    a, b = b, a
                es.add((a, b))
        return list(es)

    # cell volume deviation; perfect = 1, increasing
    def nu(self):
        areas = [poly.calculate_area() for poly in self.polygons]
        return max(areas)/min(areas)

    def face_counts(self):
        n2c = Counter()
        for poly in self.polygons:
            n2c[len(poly)] += 1
        return n2c

    def voronoi_entropy(self):
        n2c = self.face_counts()
        ve = 0
        for n in n2c:
            p = n2c[n]/len(self.polygons)
            ve -= p * math.log(p)
        return ve
    
    def mean_vertices(self):
        return sum(len(poly.vertices) for poly in self.polygons)/len(self.polygons)
    
    def angular_defect(self):
        return sum(poly.angular_defect() for poly in self.polygons)/len(self.polygons)
    
    def isoperimetric_ratio(self):
        return sum(poly.normalized_isoperimetric_ratio() for poly in self.polygons)/len(self.polygons)

    def basic_observations(self, saveto_pre, conversion=1, unit=None):
        dcount = Counter()
        for poly in self.polygons:
            dcount[len(poly.vertices)] += 1
        bins, counts = zip(*dcount.items())
        plt.figure()
        plt.bar(bins, counts)
        plt.xlabel('Number of sides')
        plt.ylabel('Count')
        plt.savefig(saveto_pre + 'nsides.png')
        
        areas = [poly.calculate_area() * conversion**2 for poly in self.polygons]
        plt.figure()
        plt.hist(areas, bins='auto')
        xlabel = 'Area'
        if unit is not None: xlabel += ' (' + unit + ')'
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.savefig(saveto_pre + 'areas.png')

    def tessellation_metrics(self):
        ve = self.voronoi_entropy()
        mv = self.mean_vertices()
        ad = self.angular_defect()
        ir = self.isoperimetric_ratio()
        nu = self.nu()
        s  = f'  Mean vertices:            {nice(mv)}\n'
        s += f'  Voronoi entropy:          {nice(ve)}\n'
        s += f'  Angular defect:           {nice(ad)}\n'
        s += f'  Isoperimetric ratio:      {nice(ir)}\n'
        s += f'  Volume deviation (nu, 1): {nice(nu)}'
        return s, (mv, ve, ad, ir, nu)
    
    def difference(self, other):
        diff = None
        total = None
        for i in range(len(self.polygons)):
            ps = shapely_of(self.polygons[i])
            if total is None: total = ps
            total = total.union(ps)
            po = shapely_of(other.polygons[i])
            total = total.union(po)
            d = ps.difference(po)
            if diff is None: diff = d
            diff = diff.union(d)
        return diff.area / total.area

    def voronoi_metrics(self, generators=None):
        if generators is None:
            generators = [poly.centroid for poly in self.polygons]
        p = self.perpendicularity(generators)
        b = self.bisectionality(generators)
        s = ''
        s += f'  Perpendicularity (1):     {nice(p)}\n'
        s += f'  Bisectionality (1):       {nice(b)}'
        return s, (p, b)

    def perpendicularity(self, generators):
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        perp, ttl = 0, 0
        for edge, polys in e2p.items():
            if len(polys) != 2: continue
            A, B = [np.array(v.coord()) for v in edge]
            C, D = [np.array(generators[p2i[poly.centroid]]) for poly in polys]
            u = (B-A) / np.linalg.norm(B-A)
            v = (D-C) / np.linalg.norm(D-C)
            perp += abs(np.dot(u, v))
            ttl += 1
        return perp/ttl
    
    def bisectionality(self, generators):
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        bi, ttl = 0, 0
        for edge, polys in e2p.items():
            if len(polys) != 2: continue
            A, B = [np.array(v.coord()) for v in edge]
            C, D = [np.array(generators[p2i[poly.centroid]]) for poly in polys]
            u = (B-A) / np.linalg.norm(B-A)
            c, d = sorted(abs(float(np.cross(E-A, u))) for E in (C, D))
            bi += (d-c)/(d+c)
            ttl += 1
        return bi/ttl

    def _components(self):
        # edge to adjacent polygons
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        # decompose into connected polygons (TODO: turn into union-find)
        poly_groups = [set([i]) for i in range(len(self.polygons))]
        pi2g = {i:i for i in range(len(self.polygons))}
        for polys in e2p.values():
            if len(polys) != 2: continue
            p, q = polys
            # poly indices
            i, j = [p2i[r.centroid] for r in polys]
            # group indices
            gi, gj = [pi2g[k] for k in (i, j)]
            if gi == gj: continue
            poly_groups[gi] = poly_groups[gi] | poly_groups[gj]
            for pi in poly_groups[gj]:
                pi2g[pi] = gi
            poly_groups[gj] = None
        poly_groups = [group for group in poly_groups if group is not None]
        for group in poly_groups:
            map_from = list(sorted(group))
            map_to = {map_from[i]:i for i in range(len(map_from))}
            yield group, map_from, map_to            
    
    def _least_squares(self, A):
        # Compute least squares nearest unit solution to Ax = 0 with
        # singular value decomposition (SVD)
        A = np.array(A)
        U, sigma, VT = np.linalg.svd(A)
        # The solution is the last column of V (or V^T in NumPy's return value)
        x = VT[-1]
        if x[0] < 0: x = -x
        return x

    def _relative_growth_rates(self, group, map_from, map_to, generators):
        # edge to adjacent polygons
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        # Build alpha * b - beta * a = 0 matrix, for growth rates 
        # alpha, beta and generator-to-generator portions a, b.
        R = []
        for edge, polys in e2p.items():
            if len(polys) != 2: continue
            assert len(polys) == 2
            # indices of each poly
            i, j = [p2i[r.centroid] for r in polys]
            if i not in group: continue
            i, j = [map_to[k] for k in (i, j)]
            # generators of each poly
            P, Q = [np.array(generators[k]) for k in (i, j)]
            # endpoints of edge
            A, B = [np.array(v.coord()) for v in edge]
            # compute lengths of projections of P, Q onto AB
            AB = B - A
            u = AB / np.linalg.norm(AB)
            projP = A + np.dot(P - A, u) * u
            projQ = A + np.dot(Q - A, u) * u
            p = np.linalg.norm(P - projP)
            q = np.linalg.norm(Q - projQ)
            # construct row for equation
            row = [0]*len(group)
            row[i] = q
            row[j] = -p
            R.append(row)
        x = self._least_squares(R)
        med = list(sorted(x))[int(len(x)/2)]
        return [v/med for v in x]

    def _inverse_rate_voronoi_model(self, group, map_from, map_to, alpha=1,
                                    bootstrap=10, pweights=None, dweights=None,
                                    stop=0.1):
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        # each row expresses (Y - X) . (B - A) / norm(B - A) = 0 for
        # generators X, Y and edge AB
        P = []  # matrix for expressing perpendicularity (dot product)
        # each row expresses (X - A) x (B - A) = -(Y - A) x (B - A)
        D = []  # matrix for expressing equal distance (cross product)
        d = []  # vector ...
        # use external generators shared across edge polygons to 
        # further constrain internal generators
        externals = [edges for edges in self._externals()
                     if p2i[e2p[self._standard_edge(edges[0])][0].centroid] in group]
        ndims = 2 * (len(group) + len(externals))
        def constrain(edge, i, j):
            nonlocal P, D, d
            A, B = [np.array(v.coord()) for v in edge]
            AB = B - A
            u = AB / np.linalg.norm(AB)
            pweight, dweight = 1, 1
            if pweights is not None:
                pweight = pweights[(i, j)]
            if dweights is not None:
                dweight = dweights[(i, j)]
            # for generators C0, C1: minimize (C1-C0) . u
            prow = [0]*ndims
            prow[2*i] = u[0] * pweight
            prow[2*i+1] = u[1] * pweight
            prow[2*j] = -u[0] * pweight
            prow[2*j+1] = -u[1] * pweight
            P.append(prow)
            # for generators C0, C1, minimize (C0-A) x u + (C1-A) x u
            drow = [0]*ndims
            drow[2*i] = u[1] * dweight
            drow[2*i+1] = -u[0] * dweight
            drow[2*j] = u[1] * dweight
            drow[2*j+1] = -u[0] * dweight
            D.append(drow)
            d.append(2 * (A[0]*u[1] - A[1]*u[0]) * dweight)
        for edge, polys in e2p.items():
            if len(polys) != 2: continue
            i, j = [p2i[poly.centroid] for poly in polys]
            if i not in group: continue
            i, j = [map_to[k] for k in (i, j)]
            constrain(edge, i, j)
        for k, edges in enumerate(externals):
            # generator j is external
            j = len(group) + k
            for edge in edges:
                poly = e2p[self._standard_edge(edge)][0]
                # generator i is internal
                i = map_to[p2i[poly.centroid]]
                constrain(edge, i, j)
        C = []  # matrix for expressing containment within polygon (cross product)
        c = []  # vector ...
        for i in group:
            poly = self.polygons[i]
            for j in range(len(poly.vertices)):
                A, B = [np.array(poly.vertices[k].coord()) for k in (j, (j+1) % len(poly.vertices))]
                e = B - A
                row = [0]*ndims
                row[2*map_to[i]] = -e[1]
                row[2*map_to[i]+1] = e[0]
                C.append(row)
                c.append(e[0]*A[1] - e[1]*A[0])
        P = np.array(P)
        D = np.array(D)
        C = np.array(C)
        x = cp.Variable(ndims)
        obj = cp.Minimize(cp.norm(P @ x, NORM)
                          + alpha * cp.norm(D @ x - d, NORM))
        con = [C @ x >= c]
        prob = cp.Problem(obj, con)
        optv = prob.solve(solver=cp.CLARABEL)
        x = x.value
        if x is None:
            print('NO SOLUTION', optv)
            raise None
        if optv > 0 and bootstrap > 0:
            new_pweights, new_dweights = {}, {}
            def compute_weights(edge, i, j):
                A, B = [np.array(v.coord()) for v in edge]
                u = (B-A)/np.linalg.norm(B-A)
                C0, C1 = [np.array(x[2*k:2*k+2]) for k in (i, j)]
                # Max weight is 1, handling cases where a generator moves close
                # to or onto an edge.
                new_pweights[(i, j)] = min(1, 1/np.linalg.norm(C1-C0))
                new_dweights[(i, j)] = min(1, 1/(sum(abs(float(np.cross(C-A, u))) for C in (C0, C1))))
            for edge, polys in e2p.items():
                if len(polys) != 2: continue
                i, j = [p2i[poly.centroid] for poly in polys]
                if i not in group: continue
                i, j = [map_to[k] for k in (i, j)]
                compute_weights(edge, i, j)
            for k, edges in enumerate(externals):
                j = len(group) + k
                for edge in edges:
                    poly = e2p[self._standard_edge(edge)][0]
                    # generator i is internal
                    i = map_to[p2i[poly.centroid]]
                    compute_weights(edge, i, j)
            recur = pweights is None
            if not recur:
                for old, new in ((pweights, new_pweights), (dweights, new_dweights)):
                    for key in old:
                        if abs(new[key] - old[key]) / old[key] > stop:
                            recur = True
                            break
                    if recur: break
            if recur:
                return self._inverse_rate_voronoi_model(group, map_from, map_to, 
                                                        alpha, bootstrap-1, 
                                                        new_pweights, new_dweights)
        return (x, optv)
    
    def _solve_irv(self, group, map_from, map_to, eps):
        x, optv = self._inverse_rate_voronoi_model(group, map_from, map_to)
        generators = [(x[2*i], x[2*i+1]) for i in range(len(group))]
        rates = self._relative_growth_rates(group, map_from, map_to, generators)
        return x, rates
    
    def inverse_rate_voronoi(self, eps=0.001):
        generators = [None]*len(self.polygons)
        rates = [None]*len(self.polygons)
        for group, map_from, map_to in self._components():
            x, grates = self._solve_irv(group, map_from, map_to, eps)
            for i in range(0, len(group)):
                generators[map_from[i]] = (x[2*i], x[2*i+1])
                rates[map_from[i]] = grates[i]
        return generators, rates
    
    def relative_growth_rates(self, generators=None):
        if generators is None:
            generators = [poly.centroid for poly in self.polygons]
        rates = [None for _ in range(len(generators))]
        for group, map_from, map_to in self._components():
            gen = [generators[i] for i in group]
            for i, r in enumerate(self._relative_growth_rates(group, map_from, map_to, gen)):
                rates[map_from[i]] = r
        return rates
    
    def neighbor_relative_growth(self, generators=None):
        if generators is None:
            generators = [poly.centroid for poly in self.polygons]
        rates = self.relative_growth_rates(generators)
        gtc = [0]*len(rates)  # greater-than count
        tn = [0]*len(rates)   # total neighbors
        e2p = self._edges_to_polygons()
        p2i = self._p2i()
        for edge, polys in e2p.items():
            if len(polys) != 2: continue
            i, j = (p2i[poly.centroid] for poly in polys)
            tn[i] += 1
            tn[j] += 1
            if rates[i] > rates[j]: gtc[i] += 1
            if rates[j] > rates[i]: gtc[j] += 1
        #return [g/t if t > 0 else 0 for g, t in zip(gtc, tn)]
        return [int(g == t) for g, t in zip(gtc, tn)]

    def perturb(self, N=1):
        edge_vtxs = set()
        # e2p = self._edges_to_polygons()
        # for edge, polys in e2p.items():
        #     if len(polys) == 1:
        #         for v in edge:
        #             edge_vtxs.add(v)
        vtxs = set()
        for poly in self.polygons:
            for vtx in poly.vertices:
                if vtx not in edge_vtxs:
                    vtxs.add(vtx)
        vtxs = list(vtxs)
        def inside(vtx):
            if len(vtx.adjacent) != 3: return True
            A, B, C = [np.array(v.coord()) for v in vtx.adjacent]
            D = np.array(vtx.coord())
            b0 = np.cross(B-A, D-A) >= 0
            b1 = np.cross(C-B, D-B) >= 0
            b2 = np.cross(A-C, D-C) >= 0
            return b0 and b1 and b2
        n = 0
        while n < N:
            vtx = random.choice(vtxs)
            if len(vtx.adjacent) != 3: continue
            p = sample_triangle(*vtx.adjacent)
            orig = vtx.coord()
            vtx.move(p[0], p[1])
            if not all(inside(adj) for adj in vtx.adjacent):
                vtx.move(orig[0], orig[1])
            else:
                n += 1
        for poly in self.polygons:
            poly.update_centroid()
        self.p2i = None
        self.e2p = None
        self.external_generators = None
                
    def random_generators(self):
        return [poly.random_coord() for poly in self.polygons]
    
    def _max_angle_distribution(self, axis, max_diams):
        angles, diams, weights, centroids = zip(*max_diams)
        circle = [2*x for x in angles]
        sweights = sum(weights)
        mcos = sum(w*math.cos(x) for x, w in zip(circle, weights))/sweights
        msin = sum(w*math.sin(x) for x, w in zip(circle, weights))/sweights
        mangle = math.atan2(msin, mcos)/2
        mr = (mcos**2 + msin**2)**0.5
        print(mangle*180/math.pi, mr)
        return angles, diams, weights, centroids

    def max_diameter_distribution(self, axis):
        max_diams = [poly.max_weighted_diameter(axis) for poly in self.polygons]
        return self._max_angle_distribution(axis, max_diams)
    
    def max_rect_distribution(self, axis):
        max_diams = [poly.max_rect(axis) for poly in self.polygons]
        #max_diams = [poly.aspect_ratio(axis) for poly in self.polygons]
        return self._max_angle_distribution(axis, max_diams)

    def max_ratio_angle_distribution(self, axis):
        max_diams = [poly.max_ratio_angle(axis) for poly in self.polygons]
        return self._max_angle_distribution(axis, max_diams)
    
    def ratio_along(self, angle):
        return sum(poly.ratio_along(angle) for poly in self.polygons)/len(self.polygons)
    
    def _vertices(self):
        return (np.array(z) for z in list(zip(*[v.coord() for poly in self.polygons for v in poly.vertices])))
    
    def contract_along(self, angle, rate):
        x, y = self._vertices()
        C = np.array((np.mean(x), np.mean(y)))
        u = np.array((math.cos(angle), math.sin(angle)))
        contracted = [poly.contract_toward(C, u, rate) 
                      for poly in self.polygons]
        return Tessellation(contracted), (C, u)
    
    def lineThrough(self, center, u):
        x, y = self._vertices()
        r0 = (min(x), min(y))
        r1 = (max(x), max(y))
        return clip_line_to_rectangle(r0, r1, center - 1e6 * u, center + 1e6 * u)
        

class VoronoiTessellation(Tessellation):
    def __init__(self, generators, boundary=None, keep=None, extra=None):
        diameter = 10**6 #np.linalg.norm(np.array(boundary.exterior.coords).ptp(axis=0))
        gen = generators
        if extra is not None:
            gen = np.concatenate((generators, extra))
        vor = Voronoi(gen, qhull_options='Qbb', incremental=False)
        ps = voronoi_polygons(vor, diameter, only_finite=(boundary is None))
        if boundary is not None:
            bps = [Polygon([Vertex(*coord) for coord in p.intersection(boundary).exterior.coords[:-1]], False) 
                   for p in ps]
        elif extra is not None:
            bps = [of_shapely(poly) for poly in list(ps)[:len(generators)]]
        else:
            bps = []
            gen = []
            for i, poly in enumerate(ps):
                if i >= len(generators): break
                bps.append(of_shapely(poly))
                gen.append(tuple(generators[i]))
            generators = gen
        generators = list(tuple(coord) for coord in generators)
        if keep is not None:
            center, radius = keep
            i = 0
            while i < len(generators):
                if distance(generators[i], center) > radius:
                    generators.pop(i)
                    bps.pop(i)
                else:
                    i += 1
        Tessellation.__init__(self, bps)
        self.generators = generators   
        self.gammas = None
                
    def _gammas(self):
        if self.gammas is None:
            self.gammas = defaultdict(lambda:1e8)
            e2p = self._edges_to_polygons()
            p2i = self._p2i()
            for adj in e2p.values():
                if len(adj) != 2: continue
                i, j = [p2i[p.centroid] for p in adj]
                d = distance(self.generators[i], self.generators[j])
                if d < self.gammas[i]: self.gammas[i] = d
                if d < self.gammas[j]: self.gammas[j] = d
        return self.gammas

    def gamma_i(self, i):
        return self._gammas()[i]
        # return min(distance(self.generators[i], self.generators[j])
        #            for j in range(len(self.generators)) if j != i)

    # aka, lambda; perfect = 0, increasing
    def cov(self):
        gammas = [self.gamma_i(i) for i in range(len(self.generators))]
        mgamma = sum(gammas)/len(gammas)
        return (sum((gamma - mgamma)**2 for gamma in gammas)/len(gammas))**(1/2) / mgamma

    # mesh ratio; perfect = 1, increasing
    def gamma(self):
        gammas = [self.gamma_i(i) for i in range(len(self.generators))]
        return max(gammas)/min(gammas)
        
    def h_i(self, i):
        return max(distance(self.generators[i], v.coord()) for v in self.polygons[i].vertices)
    
    # point distribution norm; not unitless
    def h(self):
        return max(self.h_i(i) for i in range(len(self.polygons)))

    # point distribution ratio; perfect = 1, increasing
    def mu(self):
        hs = [self.h_i(i) for i in range(len(self.polygons))]
        return max(hs)/min(hs)
    
    def chi_i(self, i):
        return 2 * self.h_i(i) / self.gamma_i(i)
    
    # regularity measure, increasing
    def chi(self):
        return max(self.chi_i(i) for i in range(len(self.polygons)))

    def regularity(self, saveto):
        regs = [self.chi_i(i) for i in range(len(self.polygons))]
        plt.figure()
        plt.hist(regs, bins='auto')
        plt.xlabel('Regularity')
        plt.ylabel('Count')
        plt.savefig(saveto)
        
    # trace of second moment tensor 
    # BUG: LIKELY NOT CORRECT
    def t_i(self, i):
        gen = np.array(self.generators[i])
        poly = self.polygons[i]
    
        # Initialize the second moment tensor
        second_moment = np.zeros((2, 2))
    
        # Iterate over the edges of the polygon
        for i in range(len(poly.vertices)):
            p1 = np.array(poly.vertices[i].coord())
            p2 = np.array(poly.vertices[(i+1)%len(poly.vertices)].coord())
    
            # Calculate the contribution of the edge to the second moment tensor
            dx, dy = p2 - p1
            a = np.dot(p1 - gen, p1 - gen)
            b = np.dot(p1 - gen, p2 - gen)
            c = np.dot(p2 - gen, p2 - gen)
            
            second_moment += np.array([[dy**2, -dx*dy], [-dx*dy, dx**2]]) * (a + b + c) / 12
    
        return np.trace(second_moment)
    
    # second moment trace measure
    def tau(self):
        ts = [self.t_i(i) for i in range(len(self.polygons))]
        mt = sum(ts)/len(ts)
        return max(abs(t - mt) for t in ts)/shapely_of(self.region()).area
    
    def distribution_metrics(self):
        cov = self.cov()
        gamma = self.gamma()
        #h = self.h()
        mu = self.mu()
        chi = self.chi()
        #tau = self.tau()
        s  = f'  COV (0):                  {nice(cov)}\n'
        s += f'  Mesh ratio (gamma, 1):    {nice(gamma)}\n'
        s += f'  Point dist ratio (mu, 1): {nice(mu)}\n'
        s += f'  Regularity (chi, 1.15):   {nice(chi)}'
        #s += f'  2nd mom tr (tau, 0):      {nice(tau)}'
        return (s, (cov, gamma, mu, chi))


def shapely_of(poly):
    coords = [v.coord() for v in poly.vertices]
    return shapely.Polygon(coords + [coords[0]])


def of_shapely(poly):
    return Polygon([Vertex(*coord) for coord in poly.exterior.coords[:-1]], True)

    
def set_op(polygons, intersection=True):
    ps = [shapely_of(poly) for poly in polygons]
    final = ps[0]
    for poly in ps[1:]:
        if intersection:
            final = final.intersection(poly)
        else:
            final = final.union(poly)
    return of_shapely(final)


def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)


def nice(x): 
    return f'{round(x, 2):.2f}'


def sample_triangle(A, B, C):
    A, B, C = [np.array(v.coord()) for v in (A, B, C)]
    u, v = B-A, C-A
    a, b = [random.uniform(0, 1) for _ in range(2)]
    if a + b > 1: a, b = 1-a, 1-b
    return A + a*u + b*v


def clip_line_to_rectangle(rect_v1, rect_v2, line_start, line_end):
    # Extract rectangle coordinates
    rx1, ry1 = rect_v1
    rx2, ry2 = rect_v2

    # Ensure rectangle coordinates are in the correct order
    if rx1 > rx2:
        rx1, rx2 = rx2, rx1
    if ry1 > ry2:
        ry1, ry2 = ry2, ry1

    # Extract line segment coordinates
    x1, y1 = line_start
    x2, y2 = line_end
    
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    slope = (y2-y1)/(x2-x1)
    
    if x1 < rx1:
        y1 += slope * (rx1 - x1)
        x1 = rx1
    
    if x2 > rx2:
        y2 += slope * (rx2 - x2)
        x2 = rx2
        
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    islope = (x2-x1)/(y2-y1)
        
    if y1 < ry1:
        x1 += islope * (ry1 - y1)
        y1 = ry1
        
    if y2 > ry2:
        x2 += islope * (ry2 - y2)
        y2 = ry2
        
    return (x1, y1), (x2, y2)


# From https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region
# Turns infinite ridges into unit vectors, then makes infinite faces 
# finite by adding points diameter units away.
def voronoi_polygons(voronoi, diameter, only_finite=False):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield shapely.Polygon(voronoi.vertices[region])
            continue
        elif only_finite:
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield shapely.Polygon(np.concatenate((finite_part, extra_edge)))


def linearish(points, lo_angle=0, hi_angle=180, precision=2, eps=0.01, reg=1.5, min_samples=5):
    # Hough transform
    angles = np.linspace(lo_angle * math.pi/180, 
                         hi_angle * math.pi/180,
                         (hi_angle - lo_angle) * precision, 
                         endpoint=False)
    x, y = (np.array(z) for z in zip(*points))
    # r = x cos theta + y sin theta
    R = np.outer(x, np.cos(angles)) + np.outer(y, np.sin(angles))
    # find clusters for each angle
    segments = set()
    used_points = set()
    for j in range(R.shape[1]):  # iterate through each column
        column = R[:, j].reshape(-1, 1)  # reshape for DBSCAN
        angle = angles[j]
        mav = max(abs(column))
        normalized = column/mav
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(normalized).labels_
        l2p = defaultdict(list)
        for i, l in enumerate(labels):
            if l != -1: l2p[l].append(i)
        # examine potental groups for roughly equal spacing
        ki = 0 if abs(math.sin(angle)) >= 1/2**0.5 else 1
        groups = [list(sorted([np.array(points[i]) for i in group], 
                              key=lambda c: c[ki])) 
                  for group in l2p.values()]
        while groups:
            group = groups.pop()
            if len(group) < min_samples: continue
            seps = [np.linalg.norm(group[i]-group[i+1]) for i in range(len(group)-1)]
            max_sep = max(seps)
            mean_rest = (sum(seps)-max_sep) / (len(seps)-1)
            if max_sep > reg * mean_rest:
                # split group at largest separation
                idx = seps.index(max_sep) + 1
                groups.append(group[:idx])
                groups.append(group[idx:])
                continue
            used_points.update(tuple(x) for x in group)
            segments.add((tuple(group[0]), tuple(group[-1])))
    return segments, used_points
        
        