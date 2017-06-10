import numpy as np
import tensorflow as tf

class Model(object):

	def __init__(self, num_classes, seq_len, word_dict_size, wv, num_filters=50, filter_size=4, emb_size=100, l2_reg_lambda = 0.01):

		tf.reset_default_graph()
		# model_path = './ckpt/lstm-cnn-att/model.ckpt'
		
		self.w  = tf.placeholder(tf.int32, [None, seq_len], name="x")
		self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
		W_emb = tf.Variable(wv,name='W_emb')
		
		# Embedding layer
		X = tf.nn.embedding_lookup(W_emb, self.w)
		
		# LSTM layer
		with tf.variable_scope('lstm'):
			lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters, state_is_tuple=True)
			lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters, state_is_tuple=True)
			
			_X = tf.unstack(X, num=seq_len, axis=1)
			outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs = _X, dtype=tf.float32)
			outputs = tf.stack(outputs, axis=1)
			X = tf.concat([X,outputs],axis=-1)
		 	h1_rnn = tf.expand_dims(X, -1)				
		
		
		# CNN+Maxpooling Layer
	
		with tf.variable_scope('cnn'):
			filter_shape = [filter_size, emb_size + 2*num_filters, 1, 2*num_filters]
			W_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")	  #Convolution parameter
			b_cnn = tf.Variable(tf.constant(0.1, shape=[2*num_filters]), name="b")		  #Convolution bias parameter
			conv = tf.nn.conv2d(h1_rnn, 
						W_cnn, 
						strides=[1, 1, 1, 1], 
						padding="VALID", 
						name="conv") 
			h1_cnn = tf.nn.relu(tf.nn.bias_add(conv, b_cnn))						
			
			##Maxpooling
			#h2_pool=tf.nn.max_pool(h1_cnn, 
			#			ksize=[1,seq_len-(filter1_size-1)-(filter2_size-1),1,1],
			#			strides=[1, 1, 1, 1],
			#			padding="VALID")
			# h2_cnn = tf.squeeze(h2_pool, axis=[1,2])

			## Attentive pooling
			W_a1 = tf.get_variable("W_a1", shape=[2*num_filters, 2*num_filters])	    	# 100x100
			tmp1 = tf.matmul(tf.reshape(h1_cnn, shape=[-1, 2*num_filters]), W_a1, name="Wy") 	# NMx100
			h2_cnn = tf.reshape(tmp1, shape=[-1, seq_len-(filter_size-1), 2*num_filters])		 		#NxMx100

			M = tf.nn.relu(h2_cnn)									# NxMx100
			W_a2 = tf.get_variable("W_a2", shape=[2*num_filters, 1]) 				# 100 x 1
			tmp3 = tf.matmul(tf.reshape(M, shape=[-1, 2*num_filters]), W_a2)  		# NMx1
			alpha = tf.nn.softmax(tf.reshape(tmp3, shape=[-1, seq_len-(filter_size-1)], name="att"))	# NxM	
			self.ret_alpha = alpha

			alpha = tf.expand_dims(alpha, 1) 						# Nx1xM
			h2_pool =  tf.matmul(alpha, h2_cnn, name="r")
			
		
		##Dropout
		h_flat = tf.reshape(h2_pool,[-1,2*num_filters])
		# h_flat = tf.reshape(h2_cnn,[-1,(seq_len-3*(filter_size-1))*2*num_filters])
		h_drop = tf.nn.dropout(h_flat,self.dropout_keep_prob)

		# Fully connetected layer
		W = tf.Variable(tf.truncated_normal([2*num_filters, num_classes], stddev=0.1), name="W")
		# W = tf.Variable(tf.truncated_normal([(seq_len-3*(filter_size-1))*2*num_filters, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
		

		l2_loss = tf.constant(0.0)
		l2_loss += tf.nn.l2_loss(W)


		# prediction and loss function
		self.predictions = tf.argmax(scores, 1)
		self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
		self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

		# Accuracy
		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))	
		self.global_step = tf.Variable(0,name='global_step',trainable=False)
		self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.loss,global_step=self.global_step)

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)
		
		self.sess.run(tf.global_variables_initializer())



	def train_step(self, W_batch, y_batch):
			feed_dict = {
				self.w 		:W_batch,
				self.dropout_keep_prob: 0.7,
				self.input_y 	:y_batch
					}
			_, step, loss, accuracy, predictions = self.sess.run([self.optimizer, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
			print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
			return step,accuracy

	def test_step(self, W_batch, y_batch):
			feed_dict = {
				self.w 		:W_batch,
				self.dropout_keep_prob:1.0,
				self.input_y :y_batch
				}
			loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict)
			return predictions
