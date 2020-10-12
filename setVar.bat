call "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

set "Model=C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader\intel\person-detection-retail-0013\FP16\person-detection-retail-0013.xml"
set Input="C:\Users\tigra\Documents\trash_img\head-pose-face-detection-female-and-male.mp4"
set Input="C:\Users\tigra\Documents\trash_img\s2.mp4"
set Input="C:\Users\tigra\Documents\trash_img\head-pose-face-detection-female.mp4"
set "fileR=C:\Users\tigra\Documents\TestProjects\PyImage\main.py -i "%Input%" -m "%Model%""
py -3.7 %fileR% 