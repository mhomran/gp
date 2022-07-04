import csv

class TagWriter:

    @staticmethod
    def write(filename, coords):
        """
        Description: Write the coordinates from a CSV file.
        Input:
            filename: the name of the CSV file.
            coords: the coordinates to be written.
        """

        with open(filename, 'w', newline="") as csvfile: 
            writer = csv.writer(csvfile)
            for coord in coords:
                writer.writerow(list(coord)) 
