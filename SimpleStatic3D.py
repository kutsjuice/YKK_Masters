# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:52:53 2021

@author: kutsj
"""

import xml.etree.ElementTree as ET
import numpy as np
import scipy.linalg as la
import time

# import matplotlib.pyplot as plt

start_time = time.time()
EPS = 1e-7

class Vertex():
    def __init__(self, x, y, z, ind):
        self.x = x
        self.y = y
        self.z = z
        
        self.ind = ind
        
        self.u = 0
        self.v = 0
        self.w = 0
        
        self.forceX = 0
        self.forceY = 0
        self.forceZ = 0
        
        self.fixedX = False
        self.fixedY = False
        self.fixedZ = False
        
        self.dirs = np.array([ind*3, ind*3+1, ind*3+2])
        
class Vol3D4V():
    def __init__(self, vi, vj, vm, vp, body):
        
        self.body = body
        self.mu = body.mu
        self.E = body.E
        self.ro = body.ro
        
        self.vertices = [vi, vj, vm, vp]
        
        vs = self.vertices
        
        V6 = la.det(np.array([[1, vi.x, vi.y, vi.z],
                              [1, vj.x, vj.y, vj.z],
                              [1, vm.x, vm.y, vm.z],
                              [1, vp.x, vp.y, vp.z]]))
        a = np.zeros(4)
        b = np.zeros(4)
        c = np.zeros(4)
        d = np.zeros(4)
        
        self.mB = np.zeros((6,12))
        
        for i in range(4):
            
            j = (i+1) % 4
            m = (i+2) % 4
            p = (i+3) % 4
            
            a[i] = la.det(np.array([[vs[j].x, vs[j].y, vs[j].z],
                                    [vs[m].x, vs[m].y, vs[m].z],
                                    [vs[p].x, vs[p].y, vs[p].z]]))
            
            b[i] = - la.det(np.array([[1, vs[j].y, vs[j].z],
                                      [1, vs[m].y, vs[m].z],
                                      [1, vs[p].y, vs[p].z]]))
            
            c[i] = - la.det(np.array([[vs[j].x, 1, vs[j].z],
                                      [vs[m].x, 1, vs[m].z],
                                      [vs[p].x, 1, vs[p].z]]))
            
            d[i] = - la.det(np.array([[vs[j].x, vs[j].y, 1],
                                      [vs[m].x, vs[m].y, 1],
                                      [vs[p].x, vs[p].y, 1]]))
            
            self.mB[:, 3*i: 3*i+3] =  \
                np.array([[b[i],    0,      0],
                          [0,       c[i],   0],
                          [0,       0,      d[i]],
                          [c[i],    b[i],   0],
                          [0,       d[i],   c[i]],
                          [d[i],    0,      b[i]]])
        
        self.mB = self.mB/V6
        
        kf1 = self.mu/(1-self.mu)
        kf2 = (1-2*self.mu)/(2*(1-self.mu))
        kf3 = self.E*(1-self.mu)/(1+self.mu)/(1-2*self.mu)
        #какойто комментарий
        self.mD = np.array ([[1,    kf1,    kf1,  0,  0,  0],
                             [kf1,  1,      kf1,  0,  0,  0],
                             [kf1,  kf1,    1  ,  0,  0,  0],
                             [0,    0,      0,    kf2,0,  0],
                             [0,    0,      0,    0,  kf2,0],
                             [0,    0,      0,    0,  0,  kf2]])
        
        self.mD = self.mD * kf3
        
        self.mK = self.mB.T @ self.mD @ self.mB * V6 / 6
        
        self.indices = np.concatenate((vi.dirs, vj.dirs, vm.dirs, vp.dirs))

        self.mM = np.array([[2,0,0,1,0,0,1,0,0,1,0,0],
                            [0,2,0,0,1,0,0,1,0,0,1,0],
                            [0,0,2,0,0,1,0,0,1,0,0,1],
                            [1,0,0,2,0,0,1,0,0,1,0,0],
                            [0,1,0,0,2,0,0,1,0,0,1,0],
                            [0,0,1,0,0,2,0,0,1,0,0,1],
                            [1,0,0,1,0,0,2,0,0,1,0,0],
                            [0,1,0,0,1,0,0,2,0,0,1,0],
                            [0,0,1,0,0,1,0,0,2,0,0,1],
                            [1,0,0,1,0,0,1,0,0,2,0,0],
                            [0,1,0,0,1,0,0,1,0,0,2,0],
                            [0,0,1,0,0,1,0,0,1,0,0,2]])
        self.mM = self.ro*V6/6/20*self.mM


    def addSelf(self, globMk, globMm):
        for i in range(self.indices.size):
            iG = self.indices[i] #номер направления в глобальной матрице жесткости
            for j in range(self.indices.size):
                jG = self.indices[j] #номер направления в глобальной матрице жесткости
                globMk[iG, jG] += self.mK[i, j]
                globMm[iG, jG] += self.mM[i, j]

    

class Body():
    def __init__(self, e, mu, ro):
        """
        Parameters
        ----------
        e : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        ro : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.vertices=[]
        self.finels=[]
        self.E = e
        self.mu = mu
        self.ro = ro
        
    def makeMesh(self, length, width, height, el_size, force):
        
        self.force = force
        nL = int(np.ceil(length/el_size))
        nW = int(np.ceil(width/el_size))
        nH = int(np.ceil(height/el_size))
        
        x_cords = np.linspace(0, length, nL+1)
        y_cords = np.linspace(0, height, nH+1)
        z_cords = np.linspace(0, width, nW+1)
        xS = x_cords[1] - x_cords[0]
        yS = y_cords[1] - y_cords[0]
        zS = z_cords[1] - z_cords[0]


        dF = self.force/(nW+1)/(nH+1)
        
        for x in x_cords[:-1]:                
            for y in y_cords[:-1]:
                for z in z_cords[:-1]:
                    v1 = self.addVertex(x, y, z)
                    v2 = self.addVertex(x+xS, y, z)
                    v3 = self.addVertex(x+xS, y, z+zS)
                    v4 = self.addVertex(x, y, z+zS)
                    
                    
                    v5 = self.addVertex(x, y + yS, z)
                    v6 = self.addVertex(x+xS, y + yS, z)
                    v7 = self.addVertex(x+xS, y + yS, z+zS)
                    v8 = self.addVertex(x, y + yS, z+zS)
                    
                    if x == 0:
                        v1.fixedX = True
                        v1.fixedY = True
                        v1.fixedZ = True
                        
                        v4.fixedX = True
                        v4.fixedY = True
                        v4.fixedZ = True
                        
                        v5.fixedX = True
                        v5.fixedY = True
                        v5.fixedZ = True
                        
                        v8.fixedX = True
                        v8.fixedY = True
                        v8.fixedZ = True
                        
                    elif x== length-xS:
                        v2.forceZ = dF
                        v3.forceZ = dF 
                        v6.forceZ = dF
                        v7.forceZ = dF
                    
                    self.addFinel(v1, v2, v4, v8)
                    self.addFinel(v2, v3, v4, v7)
                    self.addFinel(v2, v7, v4, v8)
                    
                    self.addFinel(v1, v6, v8, v5)
                    self.addFinel(v1, v2, v8, v6)
                    self.addFinel(v2, v7, v8, v6)
                        
                    
                
    

    def addFinel(self, v1, v2, v3, v4):
        el = Vol3D4V(v1, v2, v3, v4, self)
        self.finels.append(el)
        return el
    
    
    def addVertex(self, x, y, z):
        for v in self.vertices:
            if np.linalg.norm(np.array([v.x - x, v.y - y, v.z - z])) < EPS:
                return v
        
        v = Vertex(x, y, z, len(self.vertices))
        self.vertices.append(v)
        return v


    def makeMatrices(self):
        N = len(self.vertices)*3

        self.mK = np.zeros([N,N])
        self.vF = np.zeros(N)
        self.mM = np.zeros([N,N])
        
        for el in self.finels:
            el.addSelf(self.mK, self.mM)
            
        for vert in self.vertices:
            if vert.fixedX:
                self.mK[vert.dirs[0],vert.dirs[0]] +=1e9
                
            if vert.fixedY:
                self.mK[vert.dirs[1],vert.dirs[1]] +=1e9
                
            if vert.fixedZ:
                self.mK[vert.dirs[2],vert.dirs[2]] +=1e9
            
            self.vF[vert.dirs[0]] = vert.forceX
            self.vF[vert.dirs[1]] = vert.forceY
            self.vF[vert.dirs[2]] = vert.forceZ


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def createXML(body, delta):
    VTKfile = ET.Element('VTKFile', type = 'UnstructuredGrid', version='0.1', byte_order='LittleEndian')
    UnstructuredGrid = ET.Element('UnstructuredGrid')
    Piece = ET.Element('Piece', NumberOfPoints= str(len(body.vertices)) , NumberOfCells=str(len(body.finels)))
    
    PointsData = ET.Element('PointData', Vectors='delta')
    string = ''
    
    string = ''
    for i in range(int(len(delta)/3)):
        stringappend = str(delta[i*3+0]) + ' ' + str(delta[i*3+1]) + ' ' + str(delta[i*3+2]) + ' '
        string += stringappend
    DataArrayDel = ET.Element('DataArray', Name="delta", NumberOfComponents = "3", type="Float32", format="ascii")
    DataArrayDel.text = string
    PointsData.append(DataArrayDel)

    Points = ET.Element('Points')
    DataArray = ET.Element('DataArray', NumberOfComponents='3', type ='Float32', format='ascii')
    string=''
    i = 0
    for vert in body.vertices:
        string +=str(vert.x)+' '+str(vert.y)+' '+str(vert.z)+ ' '
        i +=1
    DataArray.text = string
    Points.append(DataArray)

    Cells = ET.Element('Cells')
    DataArrayCon = ET.Element('DataArray', type = 'Int32', Name ='connectivity')
    string = ''
    for fin in body.finels:
        string +=str(fin.vertices[0].ind)+ ' ' + str(fin.vertices[1].ind)+ ' ' + str(fin.vertices[2].ind)+ ' '+ str(fin.vertices[3].ind)+' '
    DataArrayCon.text = string
    Cells.append(DataArrayCon)

    
    DataArrayOffsets = ET.Element('DataArray', type = 'Int32', Name ='offsets')
    string = ''
    for i in range(1,len(body.finels)+1):
        string +=str(4*i)+' '
    DataArrayOffsets.text = string
    Cells.append(DataArrayOffsets)
    DataArrayType = ET.Element('DataArray', type = 'Int32', Name ='types')
    string = ''
    for i in range(len(body.finels)):
        string += '10 '
    DataArrayType.text = string
    Cells.append(DataArrayType)

    Piece.append(Points)
    Piece.append(PointsData)
    Piece.append(Cells)

    UnstructuredGrid.append(Piece)
    VTKfile.append(UnstructuredGrid)
    indent(VTKfile)
    etree = ET.ElementTree(VTKfile)

    myfile = open('test_1.vtu', 'wb')
    etree.write(myfile, encoding='utf-8', xml_declaration=True)
    myfile.close




def main():
    body = Body( e = 2.1e5, mu = 0.3, ro = 7900)
    body.makeMesh(length = 300,width = 50, height = 50, el_size = 20, force = 5000)
    print("%s seconds create mesh" % (time.time() - start_time))

    body.makeMatrices()
    print("%s seconds create mK" % (time.time() - start_time))

    delta = np.linalg.solve(body.mK, body.vF)
    print("%s seconds solve problem" % (time.time() - start_time))

    createXML(body,delta)
    print("%s seconds create XML" % (time.time() - start_time))



if __name__ == '__main__':
    main()
    
