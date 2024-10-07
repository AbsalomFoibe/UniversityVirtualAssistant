import numpy
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
from voc import voc  

def splitDataset(data):
    x_train = [data.getQuestionInNum(x) for x in data.questions]
    y_train = [data.getTag(data.questions[x]) for x in data.questions]
    return x_train, y_train

#Open file with correct encoding
with open("intents.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)

data = voc()

for intent in raw_data["intents"]:
    tag = intent["tag"]
    data.addTags(tag)
    for question in intent["patterns"]: 
        ques = question.lower()
        data.addQuestion(ques, tag)

x_train, y_train = splitDataset(data)
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)

# Initialising the ANN
model = models.Sequential()

# Adding first layer
model.add(layers.Dense(units=64, input_dim=len(x_train[0])))
model.add(layers.Activation('relu'))
# Adding 2nd hidden layer
model.add(layers.Dense(units=32))
model.add(layers.Activation('relu'))
# Adding output layer
model.add(layers.Dense(units=data.getTagSize()))  # Use the number of tags
model.add(layers.Activation('softmax'))

# Compiling the ANN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the ANN model to training set
model.fit(x_train, y_train, batch_size=16, epochs=150)

model.save('mymodel.keras')

# Removing questions from data as it's not needed
data.questions = {}

# Save answers from json to pickle
for intent in raw_data["intents"]:
    tag = intent["tag"]
    response = [resp for resp in intent["responses"]]
    data.addResponse(tag, response)

with open('mydata.pickle', 'wb') as handle:
    pickle.dump(data, handle)

# Predicting the test set results
x_test = numpy.array([x_train[0]])
y_pred = model.predict(x_test)
p = numpy.argmax(y_pred, axis=1)
