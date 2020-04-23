how to run python file:

python LSTM_GRU.py --rnn LSTM --trainset deep --step 18 --neurons1 50 --neurons2 50 --optimizer adam
					--loss mse --lr 0.001 --epochs 100 --batch_size 64 --save

rnn -> GRU or LSTM
trainset -> deep or shallow
step -> timestep for training
otpimizer -> adam,sgd or rmsprop
loss -> mse or msle
epochs -> can be one single epoch or list of epochs to repeat training on all of them
save -> If specified model will be save after every epoch when validation error decreases