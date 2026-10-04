"""Top-down debug rendering for analytic geometry."""
import cv2,numpy as np
def render(points,anchor,start,goal,scores,size=640):
 values=np.asarray(points+[anchor,start,goal],float);lo=values.min(0)-20;hi=values.max(0)+20;scale=(size-40)/np.maximum(hi-lo,1)
 def uv(xy):v=(np.asarray(xy)-lo)*scale;return int(v[0]+20),int(size-20-v[1])
 image=np.full((size,size,3),245,np.uint8);cv2.circle(image,uv(anchor),12,(0,0,255),-1);cv2.circle(image,uv(start),10,(255,0,0),-1);cv2.circle(image,uv(goal),10,(0,180,0),-1)
 for i,(xy,score) in enumerate(zip(points,scores)):cv2.circle(image,uv(xy),8,(0,120,255),-1);cv2.putText(image,f'{i+1}:{score:.2f}',uv(xy),0,.4,(0,0,0),1)
 cv2.line(image,uv(start),uv(anchor),(255,0,255),2);return image
