import csv

class TrackWriter:

    @staticmethod
    def write(filename, frame_id, track, detection):
        """
        Description: Write the coordinates from a CSV file.
        Input:
            filename: the name of the CSV file.
            track
        """
        
        with open(filename, 'a+',newline = '') as csvfile: 
            writer = csv.writer(csvfile)
            
            data = []
            data.append(frame_id)
            data.append(track.track_id)
            data.append(track.prediction[0][0])
            data.append(track.prediction[1][0])
            data.append(detection[0][0])
            data.append(detection[1][0])
            data.append(track.top_pos[0])
            data.append(track.top_pos[1])
            data.append(track.team)

            writer.writerow(data) 
    @staticmethod
    def initialize_file(filename):
        """
        Description: initialize the CSV file.
        Input:
            filename: the name of the CSV file.
        """
        with open(filename, 'w',newline = '') as csvfile: 
            writer = csv.writer(csvfile)
            
            data = []
            data.append('frame_id')
            data.append('track_id')
            data.append('Xpostion')
            data.append('Ypostion')
            data.append('detectionXpostion')
            data.append('detectionYpostion')
            data.append('x')
            data.append('y')
            data.append('team')

            writer.writerow(data) 