from pymel.core import *
import json
import sys.numpy 

#HOW TO GENERATE JSON DATA 
# open maya script editor and drag in this python file in the python tab at the bottom.
#IMPORTANT: all cameras must be named "camera0", "camera1", etc (do not skip a number!).
#look at bottom of the file for an example of how to generate the data

def print_camera_info(all_cams, start_frame, end_frame):
    
    data = {}
    
    #replace with a file that maya will recognize 
    output_file = '/Users/sylviebartusek/Brown_University/spring2024/csci1470/Snerfs/data/synthetic/sofa_camera_info.json'

    for curr_cam in all_cams: 
        camera = PyNode('camera' + str(curr_cam))
        print(camera)
        
        for frame in range(start_frame, end_frame + 1):
            print("frame: ", frame)
            currentTime(frame, edit=True)
            frame_number = str(frame)
            
            camera_matrix = camera.worldMatrix[0].get(t=frame)
            flattened_matrix = []
            
            for i in range(4):
                for j in range(4):
                    flattened_matrix.append(camera_matrix[i][j])

        
            id = str(curr_cam) + frame_number
            frame_data = {
                'projection_matrix': flattened_matrix,
                'focal_length': camera.getFocalLength()
            }
            data[int(id)] = frame_data
    
        

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)            
        
        # print(id + "," + str(translation[0][0]) + "," + str(translation[0][1]) + "," + str(translation[0][2]) + "," + str(rotation[0][0]) + "," + str(rotation[0][1]) + "," + str(rotation[0][2]))

#EXAMPLE: prints all info for all 8 cameras in sofa.mb
all_cameras = [0, 1, 2, 3, 4, 5, 6]
print_camera_info(all_cameras, 0, 100)



