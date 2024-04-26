import maya.cmds as cmds
import maya.OpenMaya as om
import math
import maya.cmds as cmds
import maya.OpenMayaUI as omUI

#HOW TO GENERATE DATASET
# open maya script editor and drag in this python file in the python tab at the bottom. execute the code


def print_camera_info(camera, start_frame, end_frame):
    cmds.select( 'camera' + str(camera))
    
    for f in range(start_frame, end_frame):
        
        cmds.currentTime(f, edit=True)
        
        # Get view matrix.
        view = omUI.M3dView.active3dView()
        mayaProjMatrix = om.MMatrix()
        view.projectionMatrix(mayaProjMatrix)

        matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        for x in range(0, 4):
            for y in range(0, 4):
                matrix[x][y] = round(mayaProjMatrix(x, y), 3)

        
        
        frame_number = str(f)
        if len(str(f)) == 1:
            frame_number = str(0) + str(f)
        
        id = str(camera) + frame_number
        
        print(matrix)


#EXAMPLE: prints all info for all 8 cameras in sofa.mb
#IMPORTANT: all cameras must be named "camera0", "camera1", etc (do not skip a number!).
for camera in range(7):
    print_camera_info(camera, 0, 100)


