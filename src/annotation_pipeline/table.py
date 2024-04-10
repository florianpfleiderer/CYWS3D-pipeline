''' This module reads the data from table.txt and saves into classes.
'''
import yaml
import numpy as np

class Plane:
    def __init__(self, plane: dict):
        self.center = np.array(list(plane['center'].values()))
        self.height = np.array(plane['height'])
        self.points = np.array([list(point.values()) for point in plane['points']])
        self.coeffs = np.array(list(plane['plane'].values()))
    
    def __str__(self):
        return f'center={self.center},\
            \nheight={self.height},\
            \npoints={self.points},\
            \ncoeffs={self.coeffs}'

class Table:
    def __init__(self, planes: list):
        self.planes = [Plane(plane) for plane in planes]

def read_table_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        planes = []
        while content:
            last_center_index = content.rfind('center:')
            if last_center_index == -1:
                break
            content, plane_str = content[:last_center_index], content[last_center_index:]
            plane = yaml.safe_load(plane_str)
            planes.append(plane)
            # wait = input('Press Enter to continue...')
        return Table(planes)

if __name__ == '__main__':
    table = read_table_file('../../data/annotation/office/table.txt')
    print(table.planes[0])