# coding=utf-8
import itertools
from PIL import Image
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import itchat
tf.enable_eager_execution()
count=0

itchat.auto_login(enableCmdQR=2)

def read_image(path,label):
    global count
    count+=1
    content=tf.read_file(path)
    image_tensor=tf.image.decode_jpeg(content)
    image_tensor=tf.image.per_image_standardization(image_tensor)
    image_tensor=tf.convert_to_tensor(image_tensor)
    return image_tensor,label
def read_jpg(path):
    image_path_list=[]
    label_list=[]
    single_img_path=os.walk(path).__next__()[2]
    for i,img_path in enumerate(single_img_path):
        image_path_list.append(os.path.join(path,img_path.split('.')[0]+'.jpg'))
        label_list.append(label_dict[img_path.split('.')[0]])
    return image_path_list,label_list

def read_label(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        header = next(reader)
        label_dict = {}
        for i, row in enumerate(reader):
            row_list = row[0].split(',')
            label = row_list[-1]
            img_name = "".join(row_list[0:-1])
            label_dict[img_name] = label
        return label_dict



label_dict = read_label('../input/train_labels.csv')
image_path_list,label_list = read_jpg('../input/train')
'''
def get_data(is_train):

    if is_train:
        images = []
        labels = []
        for img in image_list[0:20000]:
            images.append(img[1])
            if label_dict[img[0]]==1:
                labels.append([0,1])
            else:
                labels.append([1,0])
        return images, labels
    else:
        images = []
        labels = []
        for img in image_list[20000:20500]:
            images.append(img[1])
            if label_dict[img[0]]==1:
                labels.append([0,1])
            else:
                labels.append([1,0])
        return images, labels
'''
def get_data(is_train):
    if is_train:

        labels=[]
        for img_path in image_path_list[0:len(image_path_list)-500]:
            img_name=img_path.split('/')[-1].split('.')[0]
            if label_dict[img_name]=='1':
                labels.append([1,0])
            else:
                labels.append([0,1])

        return image_path_list[0:len(image_path_list)-500],labels
    else:
        labels=[]
        for img_path in image_path_list[len(image_path_list)-500:]:
            img_name=img_path.split('/')[-1].split('.')[0]
            if label_dict[img_name]=='1':
                labels.append([1,0])
            else:
                labels.append([0,1])
        return image_path_list[len(image_path_list)-500:],labels
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import tensorflow as tf

train_image,train_label =get_data(True)
test_image,test_label= get_data(False)
label_count = 2
print("load data complete!")

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=10):
    res = [0] * len(tensors)
    batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
    total_size = len(batch_tensors[0][1])
    batch_count = (total_size + batch_size - 1) / batch_size
    for batch_idx in range(int(batch_count)):
        current_batch_size = None
        for (placeholder, tensor) in batch_tensors:
            batch_tensor = tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            current_batch_size = len(batch_tensor)
            feed_dict[placeholder] = tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        tmp = session.run(tensors, feed_dict=feed_dict)
        res = [r + t * current_batch_size for (r, t) in zip(res, tmp)]
    return [r / float(total_size) for r in res]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current


def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in range(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')

def averate_losses(loss):
    tf.add_to_collection('losses',loss)
    losses=tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss
def average_gradients(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[g for g,_ in grad_and_vars]

        grad=tf.stack(grads,0)
        grad=tf.reduce_mean(grad,0)

        v=grad_and_vars[0][1]
        grad_and_var=(grad,v)
        average_grads.append(grad_and_var)
    return average_grads

def feed_all_gpu(inp_dict,models,payload_per_gpu,batch_x,batch_y):
    for i in range(len(models)):
        x,y=models[i]
        start_pos=i*payload_per_gpu
        stop_pos=(i+1)*payload_per_gpu
        inp_dict[x]=batch_x[start_pos:stop_pos]
        inp_dict[y]=batch_y[start_pos:stop_pos]
    return inp_dict
def run_model(image_dim, label_count, depth):
    weight_decay = 1e-4
    layers =int((depth - 4) / 3)
    graph = tf.Graph()
    with graph.as_default():

        filenames=tf.placeholder('string',shape=[None,])
        xs = tf.placeholder("float", shape=[None, 96,96,3])
        ys = tf.placeholder("float", shape=[None, label_count])
        lr = tf.placeholder("float", shape=[])

        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder("bool", shape=[])

        current = tf.reshape(xs, [-1, 96, 96, 3])
        current = conv2d(current, 3, 16, 3)

        current, features = block(current, layers, 16, 12, is_training, keep_prob)
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        current, features = block(current, layers, features, 12, is_training, keep_prob)
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        current, features = block(current, layers, features, 12, is_training, keep_prob)

        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        current = avg_pool(current, 8)
        final_dim = features*9
        current = tf.reshape(current, [-1, final_dim])
        Wfc = weight_variable([final_dim, label_count])
        bfc = bias_variable([label_count])
        ys_ = tf.nn.softmax(tf.matmul(current, Wfc) + bfc)
        cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
        correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(graph=graph,config=config) as session:
        #dataset = tf.data.Dataset.from_tensor_slices(train_image,train_label).map(read_image,num_parallel_calls=AUTOTUNE).batch(50).shuffle(500000).prefetch(buffer_size=AUTOTUNE)

        def yield_train_data():
            for image in train_image:
                yield image
        def yield_test_data():
            for image in test_image:
                yield image
        #dataset = dataset.shuffle(20500).batch(10).repeat(10)
        #dataset_t = tf.data.Dataset.from_tensor_slices((xs,ys))
        dataset = tf.data.Dataset.from_tensor_slices((filenames,ys)).map(read_image,num_parallel_calls=AUTOTUNE).shuffle(220000).batch(100).repeat(100).prefetch(buffer_size=AUTOTUNE)
        dataset_t = tf.data.Dataset.from_tensor_slices((filenames,ys)).map(read_image,num_parallel_calls=AUTOTUNE).shuffle(500).batch(100).repeat(100).prefetch(buffer_size=AUTOTUNE)
        #dataset_t = Dataset_t.shuffle(500).batch(10)
        iterator = dataset.make_initializable_iterator()
        iterator_t = dataset_t.make_initializable_iterator()
        batch_size = 64
        learning_rate = 0.1
        session.run(tf.global_variables_initializer())
        session.run(iterator.initializer,feed_dict={filenames:train_image,ys:train_label})
        session.run(iterator_t.initializer,feed_dict={filenames:test_image,ys:test_label})

        saver = tf.train.Saver()
        d = iterator.get_next()
        d_t = iterator_t.get_next()
        max_test_accuracy=0
        for epoch in range(1, 1 +300):
            if epoch == 150: learning_rate = 0.01
            if epoch == 225: learning_rate = 0.001
            for batch_idx in range(700):
                xs_, ys_ = session.run(d)
                batch_res = session.run([train_step, cross_entropy, accuracy],
                                        feed_dict={xs: xs_, ys: ys_, lr: learning_rate, is_training: True,
                                                  keep_prob: 0.8})
                if batch_idx % 100 == 0:
                    print('**---training---**')
                    print('-------------------------------')
                    print('epoch:',epoch,'/','batch_idx:', batch_idx,'/','train_step,cross_entropy,accuracy:', batch_res[1:])

            save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)

            xs_t,ys_t=session.run(d_t)
            test_results = run_in_batch_avg(session, [cross_entropy, accuracy], [xs, ys],
                                            feed_dict={xs: xs_t, ys: ys_t,
                                                       is_training: False, keep_prob:1.})

            print('**---validating---**')
            print('-------------------------------')
            print(epoch, batch_res[1:], test_results)
            if test_results[1]>max_test_accuracy and test_results[1]!=1.0:
                max_test_accuracy=test_results[1]
                time_now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                itchat.send(time_now+'\n'+'epoch: '+str(epoch)+'\n accuracy: %.4f'%max_test_accuracy,toUserName='filehelper')



def run():
    image_size = 96
    image_dim = image_size * image_size * 3

    run_model(image_dim,2,20)
    itchat.logout()


run()

