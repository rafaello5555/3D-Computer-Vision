import numpy as np

class Marker2D:
    def __init__(self,approxes,width,approximate_center):
        self.approxes = approxes.reshape((-1,2))
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.id = -1
        self.width = width
        self.side = "unknown"
        self.center_x = approximate_center[0]
        self.center_y = approximate_center[1]
        """
                E          B
                | \      / |
                |   \  /   |
                |     A    |
                |          |
                |          |
                |          |
                |          |
                |__________|
                D          C
        """
        self.assign_side()

        self.assign_points()

    def assign_side(self):
        avg_x = self.get_xavg()
        if abs(avg_x) < 50:
            self.side = "center" 
        elif avg_x < 0:
            self.side = "left"
        else:
            self.side = "right"
    
    def calculate_radius(self):
        rs = []
        for ap in self.approxes:
            r = np.sqrt((ap[0]- self.center_x)**2 + (ap[1] - self.center_y)**2)
            rs.append(r)
        return rs
    
    def assign_points(self):
        Rs = self.calculate_radius()
        N = 2
        # the farest two points
        farest = np.argpartition(Rs,kth=-N)[-N:]
        farest_points = [self.approxes[i] for i in farest]
        nearest = np.argpartition(Rs,kth=N)[:N]
        nearest_points = [self.approxes[i] for i in nearest]
        if self.side == "left":
            self.C = farest_points[0] if farest_points[0][1] > farest_points[1][1] else farest_points[1]
            self.D = farest_points[0] if farest_points[0][1] < farest_points[1][1] else farest_points[1]
            self.E = nearest_points[0] if nearest_points[0][1] < nearest_points[1][1] else nearest_points[1]
            self.B = nearest_points[0] if nearest_points[0][1] > nearest_points[1][1] else nearest_points[1]
        
            # print(self.C, self.D, "CD, left")
        
        if self.side == "right":
            self.D = farest_points[0] if farest_points[0][1] > farest_points[1][1] else farest_points[1]
            self.C = farest_points[0] if farest_points[0][1] < farest_points[1][1] else farest_points[1]
            self.B = nearest_points[0] if nearest_points[0][1] < nearest_points[1][1] else nearest_points[1]
            self.E = nearest_points[0] if nearest_points[0][1] > nearest_points[1][1] else nearest_points[1]
        
            # print(self.C, self.D, "CD, right")
            
        elif self.side == "center":
            self.assign_center_points()
    
    def assign_center_points(self):
        N = 2
        # get index of two smallest Y of approxes
        upper_indices = np.argpartition(self.approxes[:,1],kth=N)[:N]
        # get index of two largest Y of approxes
        below_indices = np.argpartition(self.approxes[:,1],kth=-N)[-N:]
        # print("indices",upper_indices, below_indices)
        below_points = self.approxes[below_indices]
        upper_points = self.approxes[upper_indices]

        # print("points",upper_points, below_points)
        self.E = upper_points[0] if upper_points[0][0] < upper_points[1][0] else upper_points[1]
        self.B = upper_points[0] if upper_points[0][0] > upper_points[1][0] else upper_points[1]
        self.D = below_points[0] if below_points[0][0] < below_points[1][0] else below_points[1]
        self.C = below_points[0] if below_points[0][0] > below_points[1][0] else below_points[1]
        # self.A = self.approxes[~np.isin(self.approxes,np.array([self.B,self.C,self.D,self.E]))]
        # BSDE = np.array([self.B,self.C,self.D,self.E])
        # self.A = result = [([i] in m[i]) for i in range(len(v))]

            
        # print(Rs, farest, farest_points,"farest")
        # far1 = self.approxes[farest[0]]
        # far2 = self.approxes[farest[1]]
        

    def get_points(self,z=False):
        if z:

            # z = np.array([np.append(self.A,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1), np.append(self.E,1)])
            if self.side == "center":
                z = np.array([np.append(self.B,1), np.append(self.C,1), np.append(self.D,1), np.append(self.E,1)])
            else:
                z = np.array([np.append(self.C,1), np.append(self.D,1)])
            return z
        else:
            # return np.array([self.B,self.C, self.D, self.E]).reshape(4,2)
            # print(f"A: {self.A}")
            # print(f"B: {self.B}")
            # print(f"C: {self.C}")
            # print(f"D: {self.D}")
            # print(f"E: {self.E}")
            # if self.side == "center":
            points =  np.array([self.B,self.C, self.D, self.E]).reshape(4,2)
            # points =  np.array([self.D])/2 + np.array([self.C]) / 2
            # else:
            #     points = np.array([self.C, self.D]).reshape(2,2)
            # print("marker points:\n", points)
            return points


    def __repr__(self):
        return self.side + " : " + str(self.B) + "\n" + str(self.C) + "\n" + str(self.D) + "\n" + str(self.E)

    def get_xavg(self):
        xs = 0
        for ap in self.approxes:
            xs += ap[0]
        return xs / 5. - self.width / 2

class Dot:
    def __init__(self,ellipse,id):
        self.id = id
        self.ellipse = ellipse
        self.x_center = 0
        self.y_center = 0
        self.assign_points()

    def assign_points(self):
        self.x_center = self.ellipse[0][0]
        self.y_center = self.ellipse[0][1]
    
    def get_id(self):
        return self.id

    def get_points(self,z=False):
        if z:
            z = np.array([self.x_center, self.y_center, 1])
            return z
        else:

            return np.array([self.x_center, self.y_center])
    

class MarkerLeft(Marker2D):
    def __init__(self, approxes, box):
        super().__init__(approxes)
        self.box = box
        """
                D          E
                | \      / |
                |   \  /   |
                |     A    |
                |          |
                |          |
                |          |
                |          |
                |__________|
                C          B
        """
    def assign_points(self):
        return super().assign_points()
    
    def get_points(self, z=False):
        if z:

            # z = np.array([np.append(self.A,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1), np.append(self.E,1)])
            z = np.array([np.append(self.E,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1)])
            # print("z",z)
            return z
        else:

            points =  np.array([self.E,self.B, self.C, self.D]).reshape(4,2)
            # print("marker points:\n", points)
            return points



class MarkerRight(Marker2D):
    def __init__(self, approxes,box):
        super().__init__(approxes)
        self.box = box
        """
                D          E
                | \      / |
                |   \  /   |
                |     A    |
                |          |
                |          |
                |          |
                |          |
                |__________|
                C          B
        """
    def assign_points(self):
        return super().assign_points()
    
    def get_points(self, z=False):
        if z:

            # z = np.array([np.append(self.A,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1), np.append(self.E,1)])
            z = np.array([np.append(self.E,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1)])
            # print("z",z)
            return z
        else:

            points =  np.array([self.E,self.B, self.C, self.D]).reshape(4,2)
            # print("marker points:\n", points)
            return points


class BEMarker():
    def __init__(self,approx):
        self.approxes = approx.reshape((-1,2))
        self.E = 0
        self.B = 0
        self.assign_points()
    
    def assign_points(self):
        N = 2
        upper_indices = np.argpartition(self.approxes[:,1],kth=N)[:N]
        upper_points = self.approxes[upper_indices]
        self.E = upper_points[0] if upper_points[0][0] < upper_points[1][0] else upper_points[1]
        self.B = upper_points[0] if upper_points[0][0] > upper_points[1][0] else upper_points[1]
        
    def get_points(self,z=False):
        if z:

            # z = np.array([np.append(self.A,1), np.append(self.B,1), np.append(self.C,1), np.append(self.D,1), np.append(self.E,1)])
            z = np.array([np.append(self.B,1), np.append(self.E,1)])
            # print("z",z)
            return z
        else:
            # return np.array([self.B,self.C, self.D, self.E]).reshape(4,2)
            # print(f"A: {self.A}")
            print(f"B: {self.B}")
            # print(f"C: {self.C}")
            # print(f"D: {self.D}")
            print(f"E: {self.E}")
            points =  np.array([self.B,self.E]).reshape(2,2)
            # print("marker points:\n", points)
            return points