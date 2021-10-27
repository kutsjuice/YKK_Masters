# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:52:53 2021

@author: kutsj
"""

import xml.etree.ElementTree as ET
import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt


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
        # print(V6)
        a = np.zeros(4)
        b = np.zeros(4)
        c = np.zeros(4)
        d = np.zeros(4)
        
        self.mB = np.zeros((6,12))
        # self.mK = np.zeros((12,12))
        
        kf1 = self.mu/(1-self.mu)
        kf2 = (1-2*self.mu)/(2*(1-self.mu))
        kf3 = self.E*(1-self.mu)/(1+self.mu)/(1-2*self.mu)

        self.mD = np.array ([[1,    kf1,    kf1,  0,  0,  0],
                             [kf1,  1,      kf1,  0,  0,  0],
                             [kf1,  kf1,    1  ,  0,  0,  0],
                             [0,    0,      0,    kf2,0,  0],
                             [0,    0,      0,    0,  kf2,0],
                             [0,    0,      0,    0,  0,  kf2]])
        
        self.mD = self.mD * kf3
        
        
        
        for i in range(4):
            
            j = (i+1) % 4
            m = (i+2) % 4
            p = (i+3) % 4
            
            # print(i, j, m, p)
            
            # a[i] = la.det(np.array([[vs[j].x, vs[j].y, vs[j].z],
            #                         [vs[m].x, vs[m].y, vs[m].z],
            #                         [vs[p].x, vs[p].y, vs[p].z]]))
            
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
            
        # for i in range(4):
        #     for j in range(4):
        #         self.mK[3*i: 3*i+3, 3*j: 3*j+3] = self.mB[:, 3*i: 3*i+3].T @ self.mD @ self.mB[:, 3*j: 3*j+3] * V6 / 6
            
        
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


    def addSelf(self):
        for i in range(self.indices.size):
            iG = self.indices[i] #номер направления в глобальной матрице жесткости
            for j in range(self.indices.size):
                jG = self.indices[j] #номер направления в глобальной матрице жесткости
                self.body.mK[iG, jG] += self.mK[i, j]
                self.body.mM[iG, jG] += self.mM[i, j]

    

class Body():
    def __init__(self, e, mu, ro):
        self.vertices=[]
        self.finels=[]
        self.E = e
        self.mu = mu
        self.ro = ro

        
    def makeMesh(self, length, width, height, el_size, force):
        
        self.force = force
        self.nL = int(np.ceil(length/el_size)) + 1 #колличество точек вдоль x
        self.nW = int(np.ceil(width/el_size)) + 1 #колличество точек вдоль y
        self.nH = int(np.ceil(height/el_size)) + 1 #колличество точек вдоль z
        
        self.x_cords = np.linspace(0, length, self.nL)
        self.y_cords = np.linspace(0, height, self.nH)
        self.z_cords = np.linspace(0, width, self.nW)

        dF = self.force/(self.nW+1)/(self.nH+1)

        self.createVertices()
        self.createFinels()
        self.createLoads()
        # self.createFixes()


    def createVertices(self):
        for z in range(self.nH):
            for y in range(self.nW):
                for x in range(self.nL):
                    ind = self.nW*z+self.nL*y+x
                    self.vertices.append(Vertex(self.x_cords[x],self.y_cords[y],self.z_cords[z],len(self.vertices)))

    def createFinels(self):
        for z in range(self.nH - 1):
            for y in range(self.nW - 1):
                for x in range(self.nL - 1):

                    v1 = self.neededVertex(0+x,0+y,0+z)
                    v2 = self.neededVertex(0+x,1+y,0+z)
                    v3 = self.neededVertex(1+x,1+y,0+z)
                    v4 = self.neededVertex(1+x,0+y,0+z)

                    v5 = self.neededVertex(0+x,0+y,1+z)
                    v6 = self.neededVertex(0+x,1+y,1+z)
                    v7 = self.neededVertex(1+x,1+y,1+z)
                    v8 = self.neededVertex(1+x,0+y,1+z)


                    self.addFinel(v5, v1, v6, v4)
                    self.addFinel(v8, v5, v6, v4)
                    self.addFinel(v8, v6, v7, v4)

                    self.addFinel(v1, v2, v6, v7)
                    self.addFinel(v1, v3, v2, v7)
                    self.addFinel(v1, v4, v3, v7)

    def neededVertex(self, x, y, z):
        ind = self.nW*self.nL*z+self.nL*y+x
        v = self.vertices[ind]
        return v

    def addFinel(self, v1, v2, v3, v4):
        el = Vol3D4V(v1, v2, v3, v4, self)
        self.finels.append(el)

    def createLoads(self):
        
        pass




    def makeMatrices(self):
        N = len(self.vertices)*3

        self.mK = np.zeros([N,N])
        self.vF = np.zeros(N)
        self.mM = np.zeros([N,N])
        
        for el in self.finels:
            el.addSelf()
            
        for vert in self.vertices:
            if vert.fixedX:
                self.mK[vert.dirs[0],vert.dirs[0]] = 1e10
                
            if vert.fixedY:
                self.mK[vert.dirs[1],vert.dirs[1]] = 1e10
                
            if vert.fixedZ:
                self.mK[vert.dirs[2],vert.dirs[2]] = 1e10
            
            self.vF[vert.dirs[0]] = vert.forceX
            self.vF[vert.dirs[1]] = vert.forceY
            self.vF[vert.dirs[2]] = vert.forceZ


class XML():

    def __init__(self, fileName, fileType, body, delta):
        self.body = body
        self.delta = delta
        self.fileName = fileName + '.' + fileType
        self.createXML()


    def createXML(self):
        etree = ET.ElementTree(self.VTK())
        myfile = open(self.fileName, 'wb')
        etree.write(myfile, encoding='utf-8', xml_declaration=True)
        myfile.close


    def VTK(self, Head = 'VTKFile', type = 'UnstructuredGrid', version ='0.1', byte_order='LittleEndian'):
        VTKfile = ET.Element( Head, type = type, version = version, byte_order= byte_order)
        VTKfile.append(self.UnstructuredGrid())
        ET.indent(VTKfile)
        return VTKfile
    
    def UnstructuredGrid(self, Head = 'UnstructuredGrid'):
        UnstructuredGrid = ET.Element(Head)
        UnstructuredGrid.append(self.Piece(NumberOfPoints = str(len(self.body.vertices)), NumberOfCells = str(len(self.body.finels))))
        return UnstructuredGrid
    
    def Piece(self, Head = 'Piece', NumberOfPoints = None, NumberOfCells = None):
        Piece = ET.Element(Head)
        if NumberOfPoints != None:
            Piece.set('NumberOfPoints', NumberOfPoints)
        if NumberOfCells != None:
            Piece.set('NumberOfCells', NumberOfCells)
        Piece.append(self.Points())
        Piece.append(self.PointData())
        Piece.append(self.Cells())
        return Piece
    
    def Points(self, Head = 'Points'):
        Points = ET.Element(Head)
        Points.append(self.DataArray(stringData = self.DataPoints(), NumberOfComponents="3", type="Float32", format="ascii"))
        return Points
    
    def PointData(self, Head = 'PointData'):
        PointsData = ET.Element(Head)
        PointsData.append(self.DataArray(stringData = self.DataDelta(), Name="delta", NumberOfComponents="3", type="Float32", format="ascii"))
        PointsData.append(self.DataArray(stringData = self.DataFixedXYZ()[0], Name="fixedX", type="Float32", format="ascii"))
        PointsData.append(self.DataArray(stringData = self.DataFixedXYZ()[1], Name="fixedY", type="Float32", format="ascii"))
        PointsData.append(self.DataArray(stringData = self.DataFixedXYZ()[2], Name="fixedZ", type="Float32", format="ascii"))
        return PointsData
    
    def Cells(self, Head = 'Cells'):
        Cells = ET.Element(Head)
        Cells.append(self.DataArray(stringData = self.DataConnectivity(), Name='connectivity',type='Int32'))
        Cells.append(self.DataArray(stringData = self.DataOffsets(), Name='offsets',type='Int32'))
        Cells.append(self.DataArray(stringData = self.DataTypes(), Name='types',type='Int32'))
        return Cells

    def DataArray(self, stringData, Head = 'DataArray', Name = None, NumberOfComponents = None, type = None, format = None):
        DataArray = ET.Element(Head)
        if Name!= None:
            DataArray.set('Name', Name)
        if NumberOfComponents!= None:
            DataArray.set('NumberOfComponents', NumberOfComponents)
        if type!= None:
            DataArray.set('type', type)
        if format!= None:
            DataArray.set('format', format)
        DataArray.text = stringData
        return DataArray
    
    def DataPoints(self):
        string=''
        for vert in self.body.vertices:
            string +=str(vert.x)+' '+str(vert.y)+' '+str(vert.z)+ ' '
            # print(vert.ind)
        return string

    def DataConnectivity(self):
        string = ''
        for fin in self.body.finels:
            string +=str(fin.vertices[0].ind)+ ' ' + str(fin.vertices[1].ind)+ ' ' + str(fin.vertices[2].ind)+ ' '+ str(fin.vertices[3].ind)+' '
        return string
    
    def DataOffsets(self):
        string = ''
        for i in range(1,len(self.body.finels)+1):
            string +=str(4*i)+' '
        return string
    
    def DataTypes(self):
        string = ''
        for i in range(len(self.body.finels)):
            string += '10 '
        return string
    
    def DataDelta(self):
        string = ''
        for i in range(int(len(self.delta)/3)):
            stringappend = str(self.delta[i*3+0]) + ' ' + str(self.delta[i*3+1]) + ' ' + str(self.delta[i*3+2]) + ' '
            string += stringappend
        return string
    
    def DataFixedXYZ(self):
        stringX = ''
        stringY = ''
        stringZ = ''
        for vert in self.body.vertices:
                if abs(vert.forceX) > 0:
                    stringX += str(1) + ' '
                else:
                    stringX += str(0) + ' '
                if abs(vert.forceY) > 0:
                    stringY += str(1) + ' '
                else:
                    stringY += str(0) + ' '
                if abs(vert.forceZ) > 0:
                    stringZ += str(1) + ' '
                else:
                    stringZ += str(0) + ' '
        return stringX, stringY, stringZ



def main():
    body = Body( e = 2.1e5, mu = 0.3, ro = 7900)
    body.makeMesh(length = 2,width = 2, height = 2, el_size = 1, force = 5000)



    print("%s seconds create mesh" % (time.time() - start_time))

    body.makeMatrices()
    print("%s seconds create mK" % (time.time() - start_time))
    
    # w, v = la.eig(body.mK@la.inv(body.mM)) #w - квадраты собственных частот; v - матрица собственных форм (каждая форма - столбец матрицы)
    # w = np.sqrt(w)
    
    

    delta = np.linalg.solve(body.mK, body.vF)
    # print(np.linalg.matrix_rank(body.mK))
    # delta = v[:,-2]
    print("%s seconds solve problem" % (time.time() - start_time))


    myXML = XML(fileName = 'mytest', fileType = 'vtu',body = body, delta = delta)

    print("%s seconds create XML" % (time.time() - start_time))
    
    # print(body.mK.shape)
    # # # print(body.mM)
    # # plt.imshow(body.mK)
    # # # plt.spy(body.mK)
    # # plt.colorbar()



if __name__ == '__main__':
    main()
    
