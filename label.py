import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

def infer(array, logits, keep_prob, image_pl, sess):
	#load model    
    image = scipy.misc.imresize(array, (256,320, 3))
   
    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
{keep_prob: 1.0, image_pl: [image]})

    #NN is less confident about classifying cars
    im_softmax0 = scipy.misc.imresize(im_softmax[0].reshape(256, 320, 3), (array.shape[0], array.shape[1]))
    binary_car_result = np.round(scipy.misc.imresize((im_softmax[0][:, 1] > float(1/3)).reshape(256, 320), (array.shape[0], array.shape[1]))/255).astype('uint8')
    #print(im_softmax[0][:, 0])
    binary_road_result = binary_car_result = np.round(scipy.misc.imresize((im_softmax[0][:, 0] > float(1/2)).reshape(256, 320), (array.shape[0], array.shape[1]))/255).astype('uint8')

    #infer model
    return [encode(binary_car_result), encode(binary_road_result)]


   

file = sys.argv[-1]

video = skvideo.io.vread(file)

answer_key = {}
hoodinfer = scipy.misc.imread('./0.png')[:,:,0]
for i in range(len(hoodinfer)-150,len(hoodinfer)):
    for j in range(len(hoodinfer[0])):
        if hoodinfer[i][j] == 10:
            hoodinfer[i][j] = 0
        else:
            hoodinfer[i][j] = 1

for i in range(0, len(hoodinfer)-150):
    for j in range(len(hoodinfer[0])):
        hoodinfer[i][j] = 1

# Frame numbering starts at 1
frame = 1
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../ToBeFrozen.ckpt.meta')    
    saver.restore(sess, '../ToBeFrozen.ckpt')
    logits = tf.reshape(tf.get_collection("finallayer")[0], (-1, 3))
    graph = tf.get_default_graph()
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    image_pl = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

        
    for rgb_frame in video:	
        # Grab red channel	
        rgb_frame[:, :, 0] = rgb_frame[:, :, 0]*hoodinfer
        answer_key[frame] = infer(rgb_frame, logits, keep_prob, image_pl, sess)    
    
        frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
