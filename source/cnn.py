import numpy as np
import tensorflow as tf

class Model(object):

	def __init__(self, num_classes, seq_len, word_dict_size, wv, num_filters=100, filter_sizes=[4,6], emb_size=100, l2_reg_lambda = 0.01):

		tf.reset_default_graph()
		# model_path = './ckpt/baselines/cnn.ckpt' 
		
		self.w  = tf.placeholder(tf.int32, [None, seq_len], name="x")
		self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
		W_emb = tf.Variable(wv,name='W_emb')
		
		# Embedding layer
		X = tf.nn.embedding_lookup(W_emb, self.w)
		X_expanded = tf.expand_dims(X,-1)
				

		# CNN+Maxpooling Layer
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			filter_shape = [filter_size, emb_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(X_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 

			# Maxpooling over the outputs
			pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			# print "pooled", pooled.get_shape				#shape=(?, 1, 1, 70)

			pooled_outputs.append(pooled)

		
		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		h_pool = tf.concat(pooled_outputs, axis=3)
		
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

		# dropout layer	 
		h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

		# Fully connetected layer
		W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
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

		# saver = tf.train.Saver()
		# save_path = saver.save(self.sess, model_path)
		# saver.restore(self.sess, model_path)
		# print("Model restored.")
		



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