# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:56:26 2023

@author: Aaron
"""

import copy
import json
import math
from matplotlib import pyplot as plt
import numpy as np
import random
import scipy
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QAction, QFileDialog, QMessageBox, QGraphicsTextItem, QDialog, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPen, QBrush, QColor, QFont, QPolygonF, QPainter

from lib import Vertex, Polygon, Tessellation, linearish


POINT_SIZE = 10
SELECTION_RADIUS = 50
EDGE_WIDTH = 5

CENTROIDS = 'Centroid'
CENTROIDS_VT = 'Centroid-generated Voronoi Diagram'
PERPENDICULARS = 'Perpendicular generators'
PERPENDICULARS_VT = 'Perpendicular Voronoi Diagram'
DIAMETERS = 'Diameters'
AXIS = 'Axis'
SEGMENTS = 'Segments'

class UITessellation:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.pointSets = {}  # List of lists of points
        self.edgeSets = {}   # List of lists of edges
        self.edgeWeights = {}
        self.areas = None
        self.axis = None

    def clear(self):
        self.vertices.clear()
        self.edges.clear()
        self.pointSets.clear()
        self.edgeSets.clear()
        self.edgeWeights.clear()
        self.areas = None
        self.axis = None
        
    def load(self, filePath):
        with open(filePath, 'r') as file:
            self.clear()
            data = json.load(file)
            self.vertices.extend(tuple(x) for x in data.get("vertices", []))
            self.edges.extend(tuple(tuple(y) for y in x) for x in data.get("edges", []))

    def loadFromTess(self, tess):
        self.clear()
        edges = tess.edges()
        self.edges.extend(edges)
        points = set()
        for edge in edges:
            points.add(edge[0])
            points.add(edge[1])
        self.vertices.extend(points)

    def save(self, filePath):
        with open(filePath, 'w') as file:
            data = {
                "vertices": self.vertices,
                "edges": self.edges
            }
            json.dump(data, file)

    def addVertex(self, coord):
        x, y = coord
        self.vertices.append(coord)
        
    def addEdge(self, edge):
        coord0, coord1 = edge
        assert coord0 in self.vertices
        assert coord1 in self.vertices
        self.edges.append(edge)

    def removeVertex(self, coord):
        if coord in self.vertices:
            self.vertices.remove(coord)
            self.edges = [edge for edge in self.edges if coord not in edge]

    def removeEdge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)
        elif (edge[1], edge[0]) in self.edges:  # Check for reversed edge
            self.edges.remove((edge[1], edge[0]))        

    def addPointSet(self, pointSet, name):
        self.pointSets[name] = pointSet

    def addEdgeSet(self, edgeSet, name, weights=None):
        self.edgeSets[name] = edgeSet
        if weights is not None:
            self.edgeWeights[name] = weights

    def removePointSet(self, name):
        if name in self.pointSets:
            del self.pointSets[name]

    def removeEdgeSet(self, name):
        if name in self.edgeSets:
            del self.edgeSets[name]

    def addAreas(self, areas):
        self.areas = areas
        
    def clearAreas(self):
        self.areas = None
        
    def resetAxis(self):
        self.axis = None
        
    def addAxisPoint(self, coord):
        if self.axis is None:
            self.axis = []
        self.axis.append(coord)


class TessellationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.tess = None
        self.pb_values = None
        self.segmentParams = [
            ('Low angle', 0),
            ('High angle', 180),
            ('Precision', 1),
            ('Epsilon', 0.006),
            ('Regularity', 1.3),
            ('Min samples', 5)]

    def initUI(self):
        # Initialize the main window and its components
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Squamation Analyzer')
        
        self.createActions()
        self.createMenus()
        self.createGraphicsView()

    def createActions(self):
        # Create actions for menu items
        self.loadImageAct = QAction('Load Image', self, triggered=self.loadImage)
        self.loadTessellationAct = QAction('Load Tessellation', self, triggered=self.loadTessellation)
        self.saveTessellationAct = QAction('Save Tessellation', self, triggered=self.saveTessellation)
        self.analyzeAct = QAction('Analyze tessellation', self, triggered=self.analyze)
        self.axisStartAct = QAction('Set axis (start)', self, triggered=self.axisStart)
        self.axisEndAct = QAction('Set axis (end)', self, triggered=self.axisEnd)
        self.analyzeOrientationAct = QAction('Analyze orientation', self, triggered=self.analyzeOrientation)
        self.analyzeOrientationsAct = QAction('Analyze orientations', self, triggered=self.analyzeOrientations)
        self.clearAnalysisAct = QAction('Clear', self, triggered=self.clearAnalysis)
        self.sampleVoronoiAct = QAction('Sample Voronoi', self, triggered=self.sampleVoronoi)
        self.sampleOrientationsAct = QAction('Sample orientations', self, triggered=self.sampleOrientations)
        self.contractAct = QAction('Contract', self, triggered=self.contractAlong)
        self.findSegmentsAct = QAction('Find segments', self, triggered=self.findSegments)
        self.regressAct = QAction('Analyze Edge', self, triggered=self.regressEdge)
        self.conversionAct = QAction('Set conversion', self, triggered=self.conversion)
        self.fullAnalysisAct = QAction('Full analysis', self, triggered=self.fullAnalysis)
        
    def createMenus(self):
        # Create menus
        self.fileMenu = self.menuBar().addMenu('File')
        self.fileMenu.addAction(self.loadImageAct)
        self.fileMenu.addAction(self.loadTessellationAct)
        self.fileMenu.addAction(self.saveTessellationAct)
        self.analyzeMenu = self.menuBar().addMenu('Analysis')
        self.analyzeMenu.addAction(self.analyzeAct)
        self.analyzeMenu.addAction(self.axisStartAct)
        self.analyzeMenu.addAction(self.axisEndAct)
        self.analyzeMenu.addAction(self.analyzeOrientationAct)
        self.analyzeMenu.addAction(self.analyzeOrientationsAct)
        self.analyzeMenu.addAction(self.sampleVoronoiAct)
        self.analyzeMenu.addAction(self.sampleOrientationsAct)
        self.analyzeMenu.addAction(self.contractAct)
        self.analyzeMenu.addAction(self.findSegmentsAct)
        self.analyzeMenu.addAction(self.clearAnalysisAct)
        self.analyzeMenu.addAction(self.conversionAct)
        self.analyzeMenu.addAction(self.fullAnalysisAct)
        self.regressMenu = self.menuBar().addMenu('Regression')
        self.regressMenu.addAction(self.regressAct)

    def createGraphicsView(self):
        # Create the QGraphicsView for image display and interaction
        self.scene = QGraphicsScene(self)
        self.view = TessellationGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if filePath:
            image = QImage(filePath)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % filePath)
                return
            pixmap = QPixmap.fromImage(image)
            self.scene.clear()  # Clear existing items in the scene
            self.pixmapItem = self.scene.addPixmap(pixmap)
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            bbox = self.scene.itemsBoundingRect()
            width, height = bbox.width(), bbox.height()
            scale = min(width, height)
            global POINT_SIZE, SELECTION_RADIUS, EDGE_WIDTH
            # ratio: 2, 10, 1
            SELECTION_RADIUS = math.ceil(scale/75)
            POINT_SIZE = math.ceil(SELECTION_RADIUS/5)
            EDGE_WIDTH = math.ceil(POINT_SIZE/2)            
            self.view.show()
        
    def hideImage(self):
        if hasattr(self, 'pixmapItem'):
            self.scene.removeItem(self.pixmapItem)
            del self.pixmapItem

    def loadTessellation(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Tessellation", "", "JSON Files (*.json)")
        if filePath:
            self.view.tess.load(filePath)
            self.view.redrawTessellation()

    def saveTessellation(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Tessellation", "", "JSON Files (*.json)")
        if filePath:
            self.view.tess.save(filePath)

    def saveSceneAsImage(self, fileName):
        image = QImage(self.view.viewport().size(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)
    
        painter = QPainter(image)
        self.view.render(painter)
        painter.end()
    
        pixmap = QPixmap.fromImage(image)
        pixmap.save(fileName, "PNG")

    def resizeEvent(self, event):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super(TessellationApp, self).resizeEvent(event)

    def closeEvent(self, event):
        QApplication.quit()
        
    def fullAnalysis(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        dialog = ParameterDialog([('N samples', 10)])
        if dialog.exec_():
            n, = dialog.values()
        full_analysis(self, n, self.view.conversion)
        
    def analyze(self, iterate=False):
        if not iterate:
            self.tess = tessellationOfEdges(self.view.tess.edges)
            mstring, pb_values = analyze(self.tess, self.view.tess)
        else:
            assert self.tess is not None
            mstring, pb_values = analyze(self.tess, self.view.tess, iterate=True)
            self.view.clearText()
        self.pb_values = pb_values
        self.view.addText(mstring, 10, 10)
        self.view.redrawTessellation()
        
    def axisStart(self):
        self.view.axisStart()
    
    def axisEnd(self):
        self.view.axisEnd()
        self.view.redrawTessellation()
        
    def conversion(self):
        dialog = ParameterDialog([('Length', 1.0)])
        if dialog.exec_():
            length, = dialog.values()
        self.view.conversionStart(length)
        
    def analyzeOrientation(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        analyze_orientation(self.tess, self.view.tess)
        self.view.redrawTessellation()

    def analyzeOrientations(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        analyze_orientations(self.tess, self.view.tess)
    
    def clearAnalysis(self):
        self.view.tess.removePointSet(CENTROIDS)
        self.view.tess.removePointSet(PERPENDICULARS)
        self.view.tess.removeEdgeSet(CENTROIDS_VT)
        self.view.tess.removeEdgeSet(PERPENDICULARS_VT)
        self.view.tess.removeEdgeSet(DIAMETERS)
        self.view.tess.removeEdgeSet(AXIS)
        self.view.tess.removeEdgeSet(SEGMENTS)
        self.view.tess.removePointSet(SEGMENTS)
        self.view.tess.clearAreas()
        self.view.clearText()
        self.view.redrawTessellation()
        
    def sampleVoronoi(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        sample_voronoi(self.tess, self.view.tess, self.pb_values)
        
    def sampleOrientations(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        sample_orientations(self.tess)

    def regressEdge(self):
        data = np.array(self.view.tess.vertices)
        h_c, k_c, r_c = circle_fit(data)
        h_h, k_h, a_h, b_h = hyperbola_fit(data)
        print(h_c, k_c, r_c)
        print(h_h, k_h, a_h, b_h)
        model = self.view.tess
        def circ(x):
            return (r_c**2 - (x-h_c)**2)**(1/2) + k_c
        def hyp(x):
            return -abs(b_h) * (1 + ((x-h_h)/a_h)**2)**(1/2) + k_h
        circle, hyperbolic = [], []
        xs = [x for x, _ in model.vertices]
        x_min, x_max = min(xs), max(xs)
        N = 100
        step = (x_max-x_min)/N
        for i in range(N+1):
            x0 = x_min + i*step
            x1 = x0 + step
            circle.append(((x0, circ(x0)), (x1, circ(x1))))
            hyperbolic.append(((x0, hyp(x0)), (x1, hyp(x1))))
        print(circle[:5])
        print(hyperbolic[:5])
        model.addEdgeSet(circle, 'Circular Arc')
        model.addEdgeSet(hyperbolic, 'Hyperbolic Arc')
        self.view.redrawTessellation()
        
    def contractAlong(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        dialog = ParameterDialog([
            ('Angle', 0.0),
            ('Rate', 0.0)])
        if dialog.exec_():
            angle, rate = dialog.values()
        self.contractAlongGiven(angle, rate)
        
    def contractAlongGiven(self, angle, rate, saveto=None):
        new_tess, (center, u) = self.tess.contract_along(angle * math.pi/180, rate)
        if saveto is not None:
            model = self.view.tess
            line = self.tess.lineThrough(center, u)
            model.addEdgeSet([line], 'Contract')
            self.view.redrawTessellation()
            self.saveSceneAsImage(saveto)
        self.tess = new_tess
        self.view.tess.loadFromTess(self.tess)
        self.view.redrawTessellation()
        
    def findSegments(self):
        self.tess = tessellationOfEdges(self.view.tess.edges)
        dialog = ParameterDialog(self.segmentParams)
        if dialog.exec_():
            self.segmentParams = [(param, value) for (param, _), value in zip(self.segmentParams, dialog.values())]
            lo, hi, precision, eps, reg, min_samples = dialog.values()
            segments, points = linearish([poly.centroid for poly in self.tess.polygons],
                                         lo_angle=lo,
                                         hi_angle=hi,
                                         precision=precision,
                                         eps=eps,
                                         reg=reg,
                                         min_samples=min_samples)
            model = self.view.tess
            model.addEdgeSet(segments, SEGMENTS)
            model.addPointSet(points, SEGMENTS)
            self.view.redrawTessellation()


class TessellationGraphicsView(QGraphicsView):
    standardColors = [QColor(255, 0, 0), QColor(0, 0, 255), QColor(0, 255, 0), QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255), QColor(128, 0, 128)]

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        super().setRenderHint(QPainter.Antialiasing)
        self.tess = UITessellation()
        self.selectedVertex = None  # Track the selected vertex for edge creation
        self.text = []
        self.analyzing = False
        self.axising = False
        self.converting = False
        self.conversion = 1
        self.conversionPoints = []
        self.scene = scene  # Set explicitly?
        self.gfxItems = []
        self.txtItems = []
        self.zoomFactor = 1.0
        self.zoomStep = 1.2

    def mousePressEvent(self, event):
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.handleLeftClick(scenePos)
        elif not self.analyzing and event.button() == Qt.RightButton:
            self.handleRightClick(scenePos)
        else:
            super().mousePressEvent(event)

    def handleLeftClick(self, scenePos):
        if self.analyzing:
            # Move generator
            for vertex in self.tess.pointSets[CENTROIDS]:
                if self.isNearVertex(scenePos, vertex):
                    if self.selectedVertex != vertex:
                        self.selectedVertex = vertex
                        return
            if self.selectedVertex is not None:
                # Maintain order of generators
                vertex = self.selectedVertex
                idx = self.tess.pointSets[CENTROIDS].index(vertex)
                self.tess.pointSets[CENTROIDS][idx] = (scenePos.x(), scenePos.y())
                self.parent().analyze(iterate=True)
        elif self.axising:
            self.tess.addAxisPoint((scenePos.x(), scenePos.y()))
        elif self.converting:
            self.conversionPoints.append((scenePos.x(), scenePos.y()))
            if len(self.conversionPoints) == 2:
                s, t = self.conversionPoints
                self.conversion /= math.hypot(s[0]-t[0], s[1]-t[1])
                self.converting = False
        else:
            # Regular mode: check if clicking near an existing vertex
            for vertex in self.tess.vertices:
                if self.isNearVertex(scenePos, vertex):
                    if self.selectedVertex and self.selectedVertex != vertex:
                        # Create or remove an edge
                        self.handleEdgeCreation(self.selectedVertex, vertex)
                        self.selectedVertex = None
                    else:
                        # Select the vertex
                        self.selectedVertex = vertex
                    return
            # No vertex selected, add a new vertex
            self.tess.addVertex((scenePos.x(), scenePos.y()))
            self.drawVertex(scenePos)

    def handleRightClick(self, scenePos):
        # Check if clicking on an existing vertex to remove it and its edges
        for vertex in self.tess.vertices:
            if self.isNearVertex(scenePos, vertex):
                self.tess.removeVertex(vertex)
                self.redrawTessellation()
                return

    def handleEdgeCreation(self, vertex1, vertex2):
        if (vertex1, vertex2) in self.tess.edges or (vertex2, vertex1) in self.tess.edges:
            self.tess.removeEdge((vertex1, vertex2))
        else:
            self.tess.addEdge((vertex1, vertex2))
        self.redrawTessellation()

    def redrawTessellation(self):
        # Remove only vertex and edge items from the scene
        for item in self.gfxItems:
            self.scene.removeItem(item)
        self.gfxItems = []
        for item in self.txtItems:
            self.scene.removeItem(item)
        self.txtItems = []

        # Draw semi-opaque areas first
        if self.tess.areas is not None:
            for poly, rate in self.tess.areas:
                self.drawPolygon(poly, rate)

        if not self.analyzing:
            for vertex in self.tess.vertices:
                self.drawVertex(QPointF(vertex[0], vertex[1]))
        for edge in self.tess.edges:
            self.drawEdge(edge[0], edge[1])
            
        # Draw additional point and edge sets
        colori = 2
        for _, pointSet in self.tess.pointSets.items():
            color = self.standardColors[colori]
            colori = (colori + 1) % len(self.standardColors)
            for point in pointSet:
                self.drawVertex(QPointF(point[0], point[1]), color, False)
        colori = 2
        for name, edgeSet in self.tess.edgeSets.items():
            color = self.standardColors[colori]
            colori = (colori + 1) % len(self.standardColors)
            weights = [1]*len(edgeSet)
            if name in self.tess.edgeWeights:
                weights = self.tess.edgeWeights[name]
            for edge, weight in zip(edgeSet, weights):
                self.drawEdge(edge[0], edge[1], color, weight)

        for txt, x, y in self.text:
            self.drawText(txt, x, y)

    def drawPolygon(self, poly, rate):
        qpoly = QPolygonF([QPointF(*v.coord()) for v in poly.vertices])
        red = int(255 * (1-rate))
        blue = int(255 * rate)
        green = min(red, blue)
        self.gfxItems.append(self.scene.addPolygon(qpoly, brush=QBrush(QColor(red, green, blue, 64))))

    def drawVertex(self, position, color=standardColors[0], select=True):
        # Calculate radius based on the current zoom level
        radius = POINT_SIZE / self.currentZoomFactor()
        self.gfxItems.append(self.scene.addEllipse(position.x() - radius, position.y() - radius, 2*radius, 2*radius, QPen(Qt.NoPen), QBrush(color)))
        if select:
            # Draw selection area
            selectionRadius = SELECTION_RADIUS / self.currentZoomFactor()
            oldAlpha = color.alpha()
            color.setAlpha(64)
            self.gfxItems.append(self.scene.addEllipse(position.x() - selectionRadius, position.y() - selectionRadius, 2*selectionRadius, 2*selectionRadius, QPen(Qt.NoPen), QBrush(color)))
            color.setAlpha(oldAlpha)

    def drawEdge(self, vertex1, vertex2, color=standardColors[1], weight=1):
        penWidth = EDGE_WIDTH / self.currentZoomFactor()
        if weight != 1:
            color = QColor(color)
            color.setAlpha(int(255*weight))
        self.gfxItems.append(self.scene.addLine(vertex1[0], vertex1[1], vertex2[0], vertex2[1], QPen(color, penWidth)))
        
    def drawText(self, txt, x, y):
        font = QFont('Courier')
        color = QColor(255, 255, 255)
        color.setAlpha(256)
        textItem = QGraphicsTextItem(txt)
        textItem.setPos(10, 10)
        textItem.setFont(font)
        textItem.setScale(3)
        textItem.setDefaultTextColor(color)
        self.scene.addItem(textItem)
        self.txtItems.append(textItem)
        
    def addText(self, txt, x, y):
        #self.text.append((txt, x, y))
        # TODO: change add/clearText to name corresponding to start/end analysis
        self.analyzing = True
        self.selectedVertex = None
        
    def clearText(self):
        self.text.clear()
        self.analyzing = False
        
    def axisStart(self):
        self.axising = True
        self.tess.resetAxis()
        
    def axisEnd(self):
        self.axising = False
        if self.tess.axis is not None:
            points = self.tess.axis
            edges = [(points[i], points[i+1]) for i in range(len(points)-1)]
            self.tess.addEdgeSet(edges, AXIS)
    
    def conversionStart(self, length):
        self.conversionPoints.clear()
        self.conversion = length
        self.converting = True 
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()
        self.redrawTessellation()

    def zoomIn(self):
        self.zoomFactor *= self.zoomStep
        self.scale(self.zoomStep, self.zoomStep)

    def zoomOut(self):
        self.zoomFactor /= self.zoomStep
        self.scale(1 / self.zoomStep, 1 / self.zoomStep)

    def currentZoomFactor(self):
        return self.zoomFactor

    def isNearVertex(self, scenePos, vertex):
        # Use Euclidean distance
        threshold = SELECTION_RADIUS / self.currentZoomFactor()  # Adjust threshold as needed
        return math.hypot(scenePos.x() - vertex[0], scenePos.y() - vertex[1]) <= threshold

    # Override resizeEvent to adjust the sizes of points and edges
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.redrawTessellation()


class ParameterDialog(QDialog):
    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Enter Parameters')
        self.layout = QVBoxLayout(self)

        self.paramEdits = []
        self.paramTypes = []
        for param, value in parameters:
            layout = QHBoxLayout()
            layout.addWidget(QLabel(param))
            self.paramEdits.append(QLineEdit(self))
            layout.addWidget(self.paramEdits[-1])
            self.paramEdits[-1].setText(str(value))
            self.paramTypes.append(type(value))
            self.layout.addLayout(layout)

        # Add a button to close the dialog
        self.submitButton = QPushButton('Submit', self)
        self.submitButton.clicked.connect(self.accept)
        self.layout.addWidget(self.submitButton)
        
    def values(self):
        return [typ(pedit.text()) for pedit, typ in zip(self.paramEdits, self.paramTypes)]


def tessellationOfEdges(edges):
    def edges_to_edges(edges):
        # return uniquified set of pairs of Vertexes, with
        # Vertexes having edges polar-sorted
        _v2v = {}
        final_edges = []
        def v2v(point):
            if point not in _v2v:
                x, y = point
                _v2v[point] = Vertex(x, y)
            return _v2v[point]
        for src, tgt in edges:
            src = v2v(src)
            tgt = v2v(tgt)
            src.add_adjacent(tgt)
            tgt.add_adjacent(src)
            final_edges.append((src, tgt))
        for v in _v2v.values():
            v.sort_adjacent_polar()
        return final_edges
    def _get_face(src : Vertex, tgt : Vertex, face):
        # traverse edges counterclockwise in tight circle
        idx = tgt.adjacent.index(src)
        next = tgt.adjacent[(idx+1)%len(tgt.adjacent)]
        if next == face[0]: return
        face.append(next)
        _get_face(tgt, next, face)
    def get_face(src, tgt):
        # get associated face for this edge
        face = [src, tgt]
        _get_face(src, tgt, face)
        return tuple(face)
    def finite(face):
        area = 0.0
        n = len(face)
        for i in range(n):
            x1, y1 = face[i].coord()
            x2, y2 = face[(i + 1) % n].coord()
            factor = x1 * y2 - x2 * y1
            area += factor
        area /= 2.0
        return area < 0
    def convex(face):
        sign = None
        n = len(face)
        for k in range(n):
            three = [face[(k+i) % n].coord() for i in range(3)]
            diffs = [[three[i+1][j]-three[i][j] for j in (0, 1)] for i in range(2)]
            cross = diffs[0][0]*diffs[1][1] - diffs[0][1]*diffs[1][0]
            this_sign = cross > 0
            if sign is None:
                sign = this_sign
            elif this_sign != sign:
                return False
        return True                
    def faces(edges):
        # find all unique faces
        _faces = set()
        final = []
        for src, tgt in edges:
            for src, tgt in ((src, tgt), (tgt, src)):
                f = get_face(src, tgt)
                if not finite(f) or not convex(f):
                    continue
                key = frozenset(f)
                if key not in _faces:
                    _faces.add(key)
                    final.append(Polygon(f))
        return final
    vedges = edges_to_edges(edges)
    fs = faces(vedges)
    return Tessellation(fs)


def growth_map(tess, model, rates):
    assert len(rates) == len(tess.polygons)
    # map rates to [0, 1]
    srates = list(sorted(rates))
    lo, mid, hi = srates[0], srates[int(len(srates)/2)], srates[-1]
    los, his = mid-lo, hi-mid
    #s = max(los, his)
    #rates = [0.5 + 0.5 * (v-mid)/s for v in rates]
    rates = [0.5 + 0.5 * (v-mid)/(los if v < mid else his) for v in rates]
    # return (poly, rate) pairs
    return [(tess.polygons[i], rates[i]) for i in range(len(rates))]


def analyze(tess, model, iterate=False):
    pg, _ = tess.inverse_rate_voronoi()
    model.addPointSet(pg, PERPENDICULARS)
    pvtess = tess.constrained_voronoi(pg)
    model.addEdgeSet(pvtess.edges(), PERPENDICULARS_VT)
    if not iterate:
        vtess = tess.constrained_voronoi()
        model.addPointSet(vtess.generators, CENTROIDS)
    else:
        vtess = tess.constrained_voronoi(model.pointSets[PERPENDICULARS])
    model.addEdgeSet(vtess.edges(), CENTROIDS_VT)
    rates = tess.relative_growth_rates()
    print(nice(min(rates)), nice(max(rates)))
    #model.addAreas(growth_map(tess, model, rates))
    #nrg = tess.neighbor_relative_growth()
    #model.addAreas(list(zip(tess.polygons, nrg)))
    # areas = np.array([poly.normalized_isoperimetric_ratio() for poly in tess.polygons])
    # areas -= min(areas)
    # areas /= max(areas)
    # model.addAreas([(poly, area) for poly, area in zip(tess.polygons, areas)])
    # TODO: select only specific metrics
    tmetrics, _ = tess.tessellation_metrics()
    vtmetrics, _ = vtess.tessellation_metrics()
    dmetrics, (_, _, _, regularity) = vtess.distribution_metrics()
    mstring = ''
    mstring += 'Base tessellation metrics:\n'
    mstring += tmetrics + '\n'
    mstring += 'Centroid-generated Voronoi tessellation metrics:\n'
    mstring += vtmetrics + '\n'
    s, (cp, cb) = tess.voronoi_metrics()
    mstring += s + '\n'
    mstring += 'Centroid distribution metrics:\n'
    mstring += dmetrics + '\n'
    #mstring += f'Overlap:                    {round(100*(1-tess.difference(vtess)), 2)}%'
    xvtmetrics, _ = pvtess.tessellation_metrics()
    xdmetrics, _ = pvtess.distribution_metrics()
    mstring += '\nVoronoi Optimized\n'
    mstring += xvtmetrics + '\n' + xdmetrics + '\n'
    s, (pp, pb) = tess.voronoi_metrics(model.pointSets[PERPENDICULARS])
    mstring += s + '\n'
    #mstring += f'Overlap:                    {round(100*(1-tess.difference(pvtess)), 2)}%'
    print(mstring)
    return mstring, (cp, cb, pp, pb, regularity)


def analyze_orientations(tess, model, saveto=None):
    (lo, hi), lo_angle, hi_angle = extremal_orientations(tess)
    print(lo_angle, lo)
    print(hi_angle, hi)
    angles = np.arange(-math.pi/2, math.pi/2, math.pi/180)
    ratios = [tess.ratio_along(angle) for angle in angles]
    max_angle = math.pi/180 * lo_angle
    plt.figure()
    plt.plot([180*angle/math.pi for angle in angles], ratios)
    plt.xlabel('Angle in degrees')
    plt.ylabel('Aspect ratio')
    if saveto is not None:
        plt.savefig(saveto)
    if saveto is None:
        analyze_orientation(tess, model, ((0, 0), (1e7*math.cos(max_angle), 1e7*math.sin(max_angle))))
    return hi_angle + 90, (hi-1)/hi


def extremal_orientations(tess, randomize=False):
    offsets = [0]*len(tess.polygons)
    if randomize:
        offsets = [random.uniform(-math.pi/2, math.pi/2) for _ in range(len(tess.polygons))]
    def f(angle, neg):
        s = sum(poly.ratio_along(angle+offset) 
                for poly, offset in zip(tess.polygons, offsets))/len(offsets)
        if neg: s = -s
        return s
    def extreme(maximize=False):
        angle = scipy.optimize.minimize(f, [0], 
                                        args=maximize, 
                                        bounds=[(-math.pi/2, math.pi/2)],
                                        method='Nelder-Mead').x # Powell?
        if type(angle) == np.ndarray:
            angle = angle[0]
        val = f(angle, maximize)
        bval = f(-math.pi/2, maximize)
        if bval < val: 
            angle = -math.pi/2
            val = bval
        if maximize: val = -val
        return angle * 180/math.pi, val
    lo_angle, lo_val = extreme(False)
    hi_angle, hi_val = extreme(True)
    return (lo_val, hi_val), lo_angle, hi_angle


def sample_orientations(tess, N=10):
    def span(randomize=False):
        ival, _, _ = extremal_orientations(tess, randomize)
        return ival[1] - ival[0]
    actual = span()
    samples = [span(True) for _ in range(N)]
    stat = Statistic('Orientation spans')
    stat.extend(samples)
    mean, std, tp, ((sws, swp), (kss, ksp)) = stat.compare(actual)
    print('SW:', sws, swp)
    print('KS:', kss, ksp)
    print('Stats:', mean, std, actual, tp)


def analyze_orientation(tess, model, axis=None):
    if axis is None: axis = model.axis
    angles, diams, weights, centroids = tess.max_rect_distribution(axis)
    angles = [180*angle/math.pi for angle in angles]
    maxw = max(weights)
    nweights = [0.1 + 0.9*w/maxw for w in weights]
    model.addEdgeSet(diams, DIAMETERS, nweights)
    plt.figure()
    if axis is not None:
        plt.subplot(211)
    plt.hist(angles, bins=17, density=True, weights=weights)
    if axis is None:
        return
    plt.subplot(212)
    side0, side1 = [], []
    A, B = (np.array(c) for c in axis)
    AB = B-A
    for angle, weight, C in zip(angles, weights, centroids):
        C = np.array(C)
        cond = np.cross(C-A, AB) >= 0
        (side0, side1)[int(cond)].append((angle, weight))
    if not side0 or not side1: return
    angles0, weights0 = zip(*side0)
    angles1, weights1 = zip(*side1)
    plt.hist(angles0, bins=17, density=True, weights=weights0, color='r', alpha=0.5)
    plt.hist(angles1, bins=17, density=True, weights=weights1, color='b', alpha=0.5)


class Statistic:
    def __init__(self, name):
        self.name = name
        self.samples = []
        
    def append(self, v):
        self.samples.append(v)

    def extend(self, vs):
        self.samples.extend(vs)

    def compare(self, measured):
        """
        Tests a sample for normality and computes the p-value for the difference
        between the sample mean and a measured value.
    
        """
        samples = np.array(self.samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        # Tests for normality:
        # Shapiro-Wilk
        normality_sw = scipy.stats.shapiro(samples)
        # Kolmogorov-Smirnov
        normality_ks = scipy.stats.kstest(samples, 'norm', args=(mean, std))
        # Perform a t-test [TODO: not yet working]
        z_score = (measured - mean) / std
        p_value = scipy.stats.norm.sf(abs(z_score))
        return (mean, std, p_value, ((n.statistic, n.pvalue) for n in (normality_sw, normality_ks)))

    def plot(self, measured, saveto=None):
        """
        Plots a histogram of sample values, the normal curve obtained from their
        mean and standard deviation, and the measured value.
        """
        samples = np.array(self.samples)
        mean = np.mean(samples)
        std = np.std(samples)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the histogram of sample values
        ax.hist(samples, bins='auto', density=True, alpha=0.6)

        # Plot the normal curve
        lo = min([samples.min()] + [v for v, _ in measured])
        hi = samples.max()
        diff = hi-lo
        lo -= 0.1 * diff
        hi += 0.1 * diff
        x = np.linspace(lo, hi, 100)
        y = scipy.stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-')

        # Plot the measured values as vertical lines
        for v, name in measured:
            ax.axvline(v, color='k', linestyle='--')
            ylim = ax.get_ylim()
            ax.text(v + 0.025*diff, ylim[0] + 0.1*(ylim[1]-ylim[0]), name)

        # Add a legend
        # ax.legend()

        # Set the plot title and axis labels
        ax.set_title(f'{self.name}')
        ax.set_yticks([])

        # Show the plot
        plt.show()
        
        if saveto is not None:
            plt.savefig(saveto)
        
    
def sample_voronoi(tess, model, pb_values, N=10, saveto_pre=None):
    cps = Statistic('Centroid Perpendicularity')
    cbs = Statistic('Centroid Bisectionality')
    ops = Statistic('Optimal Perpendicularity')
    obs = Statistic('Optimal Bisectionality')
    rps = Statistic('Random Generator Perpendicularity')
    rbs = Statistic('Random Generator Bisectionality')
    #rrps = Statistic('2xRandom perp')
    #rrbs = Statistic('2xRandom bis')
    #cdiffs = Statistic('Centroid overlap')
    #odiffs = Statistic('Opt overlap')
    #rdiffs = Statistic('Random overlap')
    regs = Statistic('Regularity')
    for n in range(N):
        #print(f'{n}/{N}')
        x = copy.deepcopy(tess)
        x.perturb(len(x.polygons) * 6 * 10)
        _, (cp, cb) = x.voronoi_metrics()
        cps.append(cp)
        cbs.append(cb)
        try:
            rpg, _ = x.inverse_rate_voronoi()
        except:
            print('Voronoi construction failure (1)')
        _, (p, b) = x.voronoi_metrics(rpg)
        ops.append(p)
        obs.append(b)
        # _, (p, b) = x.voronoi_metrics(x.random_generators())
        # rrps.append(p)
        # rrbs.append(b)
        rg = tess.random_generators()
        _, (rp, rb) = tess.voronoi_metrics(rg)
        rps.append(rp)
        rbs.append(rb)
        try:
            xcv = x.constrained_voronoi()
            _, (_, _, _, regularity) = xcv.distribution_metrics()
            regs.append(regularity)
            # cdiffs.append(1-x.difference(xcv))
            # xv = x.constrained_voronoi(rpg)
            # odiffs.append(1-x.difference(xv))
            # rv = tess.constrained_voronoi(rg)
            # rdiffs.append(1-tess.difference(rv))
        except:
            print('Voronoi construction failure (2)')
    cp, cb, pp, pb, regularity = pb_values
    # Ignoring diffs for now
    # show = [(cps, cp), (cbs, cb), (ops, pp), (obs, pb),
    #         (rps, cp), (rbs, cb), (rps, pp), (rbs, pb),
    #         (rrps, cp), (rrbs, cb), (regs, regularity)]
    show = [(cps, cp), (cbs, cb), (ops, pp), (obs, pb),
            (rps, cp), (rbs, cb), (rps, pp), (rbs, pb),
            (regs, regularity)]
    width = max(len(s.name) for s, _ in show)
    def stats(stat, measured):
        mean, std, tp, ((sws, swp), (kss, ksp)) = stat.compare(measured)
        s = stat.name + ':' + ' '*(1+width-len(stat.name))
        s += ' '.join(nice(x, s) for x, s in 
                      ((mean, False), (std, False), (measured, False), 
                       (tp, True), (sws, False), (swp, False), 
                       (kss, False), (ksp, False)))
        return s
    for (stat, measured) in show:
        print(stats(stat, measured))
    if saveto_pre is not None:
        show = [(cps, [(cp, 'CP')], 'centroids_perp'),
                (cbs, [(cb, 'CB')], 'centroids_bis'),
                (ops, [(pp, 'OP')], 'opt_perp'),
                (obs, [(pb, 'OB')], 'opt_bis'),
                (rps, [(cp, 'CP'), (pp, 'OP')], 'random_perp'),
                (rbs, [(cb, 'CB'), (pb, 'OB')], 'random_bis'),
                (regs, [(regularity, 'R')], 'regularity')
                ]
        for stat, measured, name in show:
            stat.plot(measured, saveto_pre + name + '.png')

def circle_fit(data):
    # Initial guess for circle parameters
    x_m = np.mean(data[:, 0])
    y_m = np.mean(data[:, 1])
    initial_guess = [x_m, y_m, 1]
    
    # Define the cost function (sum of squared distances)
    def cost(params):
        h, k, r = params
        return sum((np.sqrt((x - h)**2 + (y - k)**2) - r)**2 for x, y in data)
    
    # Minimize the cost function
    result = scipy.optimize.minimize(cost, initial_guess, method='SLSQP')
    return result.x  # Returns (h, k, r)


def hyperbola_fit(data):
    # Initial guess for hyperbola parameters
    x_m = np.mean(data[:, 0])
    y_m = np.max(data[:, 1])
    initial_guess = [x_m, y_m, -1, 1]

    # Define the cost function
    def cost(params):
        h, k, a, b = params
        return sum((-(x - h)**2 / a**2 + (y - k)**2 / b**2 - 1)**2 for x, y in data)

    # Minimize the cost function
    result = scipy.optimize.minimize(cost, initial_guess, method='trust-constr') #, method='SLSQP')
    return result.x  # Returns (h, k, a, b)


def nice(x, sci=False): 
    if not sci: return f'{round(x, 2):.2f}'
    return f'{x:.3e}'


def full_analysis(app, N=100, conversion=1):
    app.clearAnalysis()
    
    # initial: mosaic
    # figure: mosaic
    app.saveSceneAsImage('figs/initial.png')
    
    # for the remaining figures, don't show figure
    app.hideImage()

    # first analysis: cb, cp, pp, pb, regularity
    # figures: mosaics
    app.analyze()
    cp, cb, pp, pb, regularity = app.pb_values
    #app.saveSceneAsImage('figs/original.png')
    #app.clearAnalysis()

    # basic data: polygon degrees and areas
    unit = None
    if conversion != 1: unit = r'$\mathrm{cm}^2$'
    app.tess.basic_observations('figs/basic_', conversion, unit)
    
    # orientation and contraction
    # figures: aspect ratio curve, contraction line
    angle, rate = analyze_orientations(app.tess, app.view.tess, 'figs/aspect.png')
    app.contractAlongGiven(angle, rate, 'figs/contraction_line.png')
        
    # second analysis: cb, cp, pp, pb, regularity
    # figures: mosaics without photo
    app.analyze()
    ccp, ccb, cpp, cpb, cregularity = app.pb_values
    app.saveSceneAsImage('figs/contracted.png')
    app.clearAnalysis()
    
    # regularity
    vtess = app.tess.constrained_voronoi()
    vtess.regularity('figs/basic_regularity.png')
    
    # stats on contracted cb, cp, pp, pb, regularity
    # figures: for each stat, histogram with fitted normal curve and value
    if N > 0:
        sample_voronoi(app.tess, app.view.tess, (ccp, ccb, cpp, cpb, cregularity),
                       N, 'figs/stats_')
    
    plt.close('all')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TessellationApp()
    ex.show()
    sys.exit(app.exec_())