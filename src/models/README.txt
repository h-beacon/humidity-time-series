How to run python file.
Example:

python LSTM_GRU.py --rnn lstm --trainset deep --step 18 --layers 2 --neurons 50 --optimizer adam
					--loss mse --lr 0.01 --epochs 100 --batch_size 64 --save

rnn -> GRU or LSTM
trainset -> deep or shallow
step -> timestep for training
layers -> number of LSTM/GRU layers
neurons -> number of neurons on each layer
otpimizer -> adam, sgd or rmsprop
loss -> mse or msle
epochs -> can be one single epoch or list of epochs to repeat training on all of them
save -> If specified model will be save after every epoch when validation error decreases