from pydoc import visiblename
import cv2

def get_frames(output_folder, fps, source_file = ''):
    if source_file:
        vc = cv2.VideoCapture(source_file)
        camera_flag = False
    else:
        vc = cv2.VideoCapture(0)
        camera_flag = True

    capture = 1
    i = 0
    if vc.isOpened(): #check if video opened
        rval, videoFrame = vc.read()
    else:
        rval = False
        print("Failed")
        return
    if camera_flag:
        while True:   #擷取視頻至結束
            rval, videoFrame = vc.read()
            if not rval:
                print("Failed")
                return
            if(capture % fps == 0): #每隔幾幀進行擷取
                #videoImages.append(videoFrame)
                cv2.imwrite(output_folder+"frame"+str(i)+".png", videoFrame)
                i += 1
            capture += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        while rval:   #read until video ended
            rval, videoFrame = vc.read()
            if(capture % fps == 0): #capture pic every (fps) frame 
                #videoImages.append(videoFrame) 
                cv2.imwrite(output_folder+"frame"+str(i)+".png", videoFrame)
                i += 1
            capture += 1

if __name__ == '__main__':
    output_folder = "images/real_frames/"
    fps = 30
    videoName = 'test.mp4'
    get_frames(output_folder=output_folder, fps=fps, source_file=videoName)