
# coding: utf-8

# In[43]:


import tensorflow as tf

# Model Parameter (W*x + b)
W = tf.Variable([.3] , dtype=tf.float32)
b = tf.Variable([-.3] , dtype=tf.float32)

# Input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


# In[44]:


linear_model = W*x + b

#Estimated Error using Squared error method
error = tf.reduce_sum(tf.square((linear_model -y)))


# In[45]:


#optimize using gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error)


# In[46]:


#Training Data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]


# In[47]:


#Taining
# Initialize variable and reset all
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Now Let's start the training

for i in range(1000): #Run 1000 iterratins for better accuracy
    sess.run(train , {x:x_train , y:y_train})


# In[48]:


#Check the accuracy of the trining.
recent_W , recent_b , recent_error = sess.run([W,b,error] , {x:x_train,y:y_train})
print("W: %s b: %s loss: %s"%(recent_W, recent_b, recent_error))

