package main

import (
	"fmt"
	"github.com/andrewsmedina/ultn/datasets/mnist"
	"github.com/andrewsmedina/ultn/models"
	"github.com/andrewsmedina/ultn/layers"

)
// import keras
// from keras.datasets import mnist
// from keras.models import Sequential
// from keras.layers import Dense, Dropout, Flatten
// from keras.layers import Conv2D, MaxPooling2D

const (
	batchSize int = 128
	numClasses int = 10
	epochs int = 12
)

func main() {
	imgRows := 28
	imCols := 28


	xTrain, yTrain, xTest, yTest := mnist.loadData()
	xTrain := xTrain.reshape(xTrain.shape[0], imgRows, imgCols, 1)
	xTest := xTest.reshape(xTest.shape[0], imgRows, imgCols, 1)

	inputShape := []int{img_rows, img_cols, 1}

	// x_train = x_train.astype("float32")
	// x_test = x_test.astype("float32")
	// x_train /= 255
	// x_test /= 255

	fmt.Println("x_train shape:", xTrain.shape)
	fmt.Println(xTrain.shape[0], "train samples")
	fmt.Println(xTest.shape[0], "test samples")

	// convert class vectors to binary class matrices
	yTrain = keras.utils.toCategorical(yTrain, numClasses)
	yTest = keras.utils.toCategorical(yTest, numClasses)

	model := Sequential()
	model.add(Conv2D(32, kernelSize=(3, 3),
					activation="relu",
					inputShape=inputShape))
	model.add(Conv2D(64, (3, 3), activation="relu"))
	model.add(MaxPooling2D(poolSize=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(numClasses, activation="softmax"))

	model.compile(loss=keras.losses.categoricalCrossentropy,
				optimizer=keras.optimizers.Adadelta(),
				metrics=["accuracy"])

	model.fit(xTrain, yTrain,
			batch_size=batchSize,
			epochs=epochs,
			verbose=1,
			validationData=(xTest, yTest))
	score = model.evaluate(xTest, yTest, verbose=0)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])
}
