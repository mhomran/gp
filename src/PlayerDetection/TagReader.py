import csv

class TagReader:

    @staticmethod
    def read(filename):
        """
        Description: Read the coordinates from a CSV file.
        Input:
            filename: the name of the CSV file.
        Output:
            coords: the read coordinates.
        """

        coords = []
        with open(filename, mode ='r')as file: 
            reader = csv.reader(file) 
            for line in reader: 
                x = tuple(map(int, line))
                coords.append(x)

        return coords
